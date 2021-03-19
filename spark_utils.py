import inspect
from datetime import datetime
import logging
import os
import ast
import pandas as pd
from collections import OrderedDict
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as psf
from pyspark.sql.window import Window
from pyspark.sql.types import (StringType,
                               StructType,
                               StructField,
                               DoubleType,
                               LongType,
                               IntegerType,
                               BooleanType,
                               ArrayType,
                               DateType)

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true")


def pipe(self, func, *args, **kwargs):
    return func(self, *args, **kwargs)


def value_counts(self, groupby_columns):
    return self.groupby(groupby_columns).count().orderBy(psf.col('count').desc())


DataFrame.pipe = pipe
DataFrame.value_counts = value_counts


def to_csv(self, output_path):
    """ Write dataframe to HDFS as a single csv output file.
    First file is stored on temporary directory, then single partition file name is found
    and finally moved to the target destination.

    Parameters
    -------
    self: pyspark.sql.dataframe.DataFrame
    output_path: str

    Returns
    -------
    None
    """
    output_dir, file_name = '/'.join(output_path.split('/')[:-1]), output_path.split('/')[-1]
    temp_dir_path = output_dir + '/temp'
    (self.repartition(1)
     .write
     .mode("overwrite")
     .options(header=True, delimiter=';', encoding='utf-8')
     .csv(temp_dir_path))

    sc = spark.sparkContext
    path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    filesystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hdfs_file_system = filesystem.get(sc._jsc.hadoopConfiguration())

    part_file_format = 'csv'
    part_file_path = path(temp_dir_path + f'/part*.{part_file_format}')
    part_file_name = hdfs_file_system.globStatus(part_file_path)[0].getPath().getName()

    src = path(temp_dir_path + f'/{part_file_name}')
    dest = path('.'.join([output_path, part_file_format]))

    if hdfs_file_system.exists(dest):
        hdfs_file_system.delete(dest, True)

    hdfs_file_system.rename(src, dest)


DataFrame.to_csv = to_csv


class WindowAggregator:
    """ Used to simplify window calculations and automate column naming.
    The reason why window aggregations are preferred to joins is because input table is accessed only once and
    partition calculations can be distributed across executors."""

    def __init__(self, aggregation_name, aggregation_column, partition_by, order_by=None, window_length=None):
        """

        Parameters
        ----------
        aggregation_name: str
        aggregation_column: str
        partition_by: list
        order_by: str
            optional, if not provided window is created only on partition columns
        window_length: int
            optional, if provided then rolling aggregation is calculated
        """
        self.aggregation_name = aggregation_name
        self.partition_by = partition_by
        self.order_by = order_by
        self.aggregation_column = aggregation_column
        self.window_length = window_length

    @property
    def column_alias(self):
        """ Create column name in the following format:
        <partition_columns>__<aggregation_column>__<aggregation_name>
        or
        <partition_columns>__<aggregation_column>__<aggregation_name>__<window_length>days,
        if window length is specified

        Parameters
        ----------

        Returns
        -------
        str
        """
        alias = self.partition_by + [self.aggregation_column, self.aggregation_name]
        alias = alias + [str(self.window_length) + 'days'] if self.window_length else alias
        return '__'.join(alias)

    @staticmethod
    def get_window(partition_by, order_by=None, window_length=None):
        """ Construct window object for aggregations and feature creation.
        If window length is passed, sort column is assumed to be date
        and is cast into seconds to allow specifying rangeBetween for rolling window.

        Parameters
        ----------
        partition_by: list
        order_by: str
        window_length: int
            Window length going back in days

        Returns
        -------
        pyspark.sql.window.WindowSpec
        """
        window = Window.partitionBy(partition_by)

        if order_by is not None:
            order_by_col = psf.col(order_by)
            if window_length is not None:
                order_by_col = order_by_col.cast('timestamp').cast('long')
                days = lambda i: i * 86400
                window = window.orderBy(order_by_col).rangeBetween(-days(window_length), 0)
            else:
                window = window.orderBy(order_by_col)

        return window

    def transform(self, df):
        """ Applies aggregation specified in instance to dataframe.
        If aggregation is not a method of this class, then it is searched for in pyspark.sql.functions

        Parameters
        ----------
        df: pyspark.sql.dataframe.DataFrame

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        if hasattr(self, self.aggregation_name):
            aggregation = getattr(self, self.aggregation_name)
            return aggregation(df)
        else:
            aggregation = getattr(psf, self.aggregation_name)
            w = self.get_window(self.partition_by, self.order_by, self.window_length)
            return df.withColumn(self.column_alias, aggregation(self.aggregation_column).over(w))

    def nunique(self, df):
        """ Calculates number of unique values in a column over a window"""
        w = self.get_window(self.partition_by, self.order_by, self.window_length)
        return df.withColumn(self.column_alias, psf.size(psf.collect_set(self.aggregation_column).over(w)))

    def median(self, df):
        """ Calculates median using pyspark.sql inbuilt percentile_approx method"""
        w = self.get_window(self.partition_by, self.order_by, self.window_length)
        return df.withColumn(self.column_alias, psf.expr(f'percentile_approx({self.aggregation_column}, 0.5)').over(w))

    def days_between(self, df):
        """ Calculates days between subsequent instances, aggregation column should be date.
        Note usage of a second window to fill days between values for the same date."""
        w = self.get_window(self.partition_by, self.order_by)
        w2 = self.get_window(partition_by=self.partition_by + [self.order_by],
                             order_by=self.order_by)

        last_date = psf.lag(self.aggregation_column).over(w)
        last_not_equal_current = last_date != psf.col(self.aggregation_column)
        days_between = psf.datediff(self.aggregation_column, last_date)

        df = df.withColumn(self.column_alias, psf.when(last_not_equal_current, days_between))
        return df.withColumn(self.column_alias, psf.max(self.column_alias).over(w2))

    def percent_last_total(self, df):
        """ First calculates totals on, e.g. customer-month level,
        Then adds row number to identify when month changes.
        Lagged values from previous month are calculated and filled in from first row number to entire month.
        Finally, a ratio of current value relative to previous period total is found."""
        partition_by = self.partition_by + [self.order_by]
        df = df.withColumn('dummy', psf.lit(1))
        w = self.get_window(partition_by=partition_by, order_by='dummy')
        w2 = self.get_window(partition_by=self.partition_by, order_by=self.order_by)

        period_total = psf.sum(self.aggregation_column).over(w)
        row_number = psf.row_number().over(w)
        last_period_total = psf.max(psf.when(row_number == 1, psf.lag(period_total).over(w2))).over(w)
        df = df.withColumn(self.column_alias, psf.col(self.aggregation_column) / last_period_total)
        return df.drop('dummy')

    def __repr__(self):
        return (f'Transform: {self.__class__.__name__}, \n'
                f'Parameters: \n'
                f'  aggregation_column - {self.aggregation_column} \n'
                f'  aggregation_name - {self.aggregation_name} \n'
                f'  partition columns: {self.partition_by} \n'
                f'  sort column: {self.order_by} \n'
                f'  window_lengths - {self.window_length} \n')

import ast
import pandas as pd
import os
import inspect
from pyspark.sql.types import (StringType,
                               StructType,
                               StructField,
                               DoubleType,
                               LongType,
                               IntegerType,
                               BooleanType,
                               ArrayType,
                               DateType)

from collections import OrderedDict
import numpy as np


class ReadCsvHelper:

    @classmethod
    def _get_csv_paths_related_to_function(cls, caller_name, caller_filename):
        """ Only supposed to be called by testing functions and helps to automatically paths to csv files.
         Assumes that testing functions has respective input/expected csv files in test_data/<module_name>
         folder named <test_function_name>_<argument_name>"""
        caller_module = caller_filename.split('/')[-1].split('.')[0]
        path_to_data = os.path.join(os.path.dirname(caller_filename), f'test_data/{caller_module}')
        return [os.path.join(path_to_data, filename) for filename in os.listdir(path_to_data)
                if caller_name in filename]

    @classmethod
    def get_input_dict_expected_df(cls, spark_session):
        caller_frame = inspect.currentframe().f_back.f_code
        caller_name, caller_filename = caller_frame.co_name, caller_frame.co_filename

        csv_path_URIs = cls._get_csv_paths_related_to_function(caller_name, caller_filename)
        dfs = cls.get_dfs_from_csv_path_URIs(spark_session, csv_path_URIs)

        extract_kwarg_name = lambda path: path.split('.')[0].split('/')[-1].replace(f'{caller_name}_', '')
        input_dict = {
            extract_kwarg_name(csv_path): df
            for csv_path, df in zip(csv_path_URIs, dfs)
        }
        expected_df = input_dict.pop('expected')

        return input_dict, expected_df

    @classmethod
    def get_dfs_from_csv_path_URIs(cls, spark_session, csv_path_URIs):
        return [cls.get_df_from_csv(spark_session, csv_path_URI) for csv_path_URI in csv_path_URIs]

    @classmethod
    def get_df_from_csv(cls, spark_session, csv_file_path):
        general_schema = cls._get_general_schema(csv_file_path)
        pandas_schema = cls._get_schema_for_dfp(general_schema)

        dfp = (pd.read_csv(csv_file_path,
                           sep=',',
                           keep_default_na=False,
                           na_values=['-1.#IND', '1.#QNAN', '1.#IND',
                                      '-1.#QNAN', '#N/A', 'N/A', '#NA', 'NA', 'NaN', '-NaN', 'nan', '-nan'],
                           dtype=pandas_schema,
                           skipinitialspace=True,
                           usecols=pandas_schema.keys(),
                           skiprows=1)
               .replace({np.nan: None})
               .pipe(cls._evaluate_array_columns, general_schema=general_schema))

        spark_schema = cls._get_schema_for_df(general_schema)
        return spark_session.createDataFrame(dfp, schema=spark_schema)

    @classmethod
    def _get_general_schema(cls, csv_file_path):
        datatypes, columns = pd.read_csv(csv_file_path, nrows=2, header=None, skipinitialspace=True).values
        return OrderedDict((column, datatype) for column, datatype in zip(columns, datatypes) if column != 'comment')

    @classmethod
    def _get_schema_for_dfp(cls, general_schema):
        return OrderedDict((column, cls._str_to_pandas_datatype(datatype))
                           for column, datatype in general_schema.items())

    @classmethod
    def _evaluate_array_columns(cls, dfp, general_schema):
        def _literal_eval(x):
            try:
                return ast.literal_eval(x)
            except ValueError:
                return None

        array_columns = [column for column, datatype in general_schema.items() if datatype == 'arr']
        for array_column in array_columns:
            dfp[array_column] = dfp[array_column].apply(_literal_eval)
        return dfp

    @classmethod
    def _get_schema_for_df(cls, general_schema):
        return StructType([StructField(column, cls._str_to_pyspark_datatype(datatype), True)
                           for column, datatype in general_schema.items()])

    @staticmethod
    def _str_to_pandas_datatype(datatype):
        mapper = {
            'str': str,
            'int32': int,
            'int64': int,
            'float': float,
            'boolean': bool,
            'arr': object
        }
        return mapper[datatype]

    @staticmethod
    def _str_to_pyspark_datatype(datatype):
        mapper = {
            'str': StringType(),
            'int64': LongType(),
            'int32': IntegerType(),
            'float': DoubleType(),
            'boolean': BooleanType(),
            'arr': ArrayType(elementType=StringType())
        }
        return mapper[datatype]
