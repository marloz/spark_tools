import inspect
from datetime import datetime
import logging
import os

import tools.general as tg
import tools.utils as tu

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as psf

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true")


def pipe(self, func, *args, **kwargs):
    return func(self, *args, **kwargs)


def value_counts(self, groupby_columns):
    return self.groupby(groupby_columns).count().toPandas().sort_values('count', ascending=False)


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
        e.g. '/projects/36px_retail_inflow_classifier/gpu/data/ic_model_features'

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


def load_config(config_path):
    return tg.dotdict(tu.ConfigurationLoader().load_config(config_path))


def generate_table_name(config, project_name='gpu'):
    caller_module = inspect.currentframe().f_back.f_code.co_filename.split('/')[-1].split('.')[0]
    experiment_name = config.control['experiment_name']
    return f'{project_name}___{caller_module}___{experiment_name}'


def log_decorator(log_on_start=None, log_on_end=None):
    """ Log decorator to use on class methods, assuming that class has logger attribute.
    Logs info about method arguments before or after its execution"""

    def deco(func):
        def wrapper(*args, **kwargs):
            logger = args[0].logger
            if log_on_start:
                logger.info(log_on_start.format(**kwargs))
            output = func(*args, **kwargs)
            if log_on_end:
                logger.info(log_on_end.format(**kwargs))
            return output

        return wrapper

    return deco


def add_run_specifications(df,
                           run_date=datetime.now().strftime('%Y-%m-%d %HH:%MM:%SS'),
                           git_version=None):
    git_version = git_version if git_version else tu.get_git_version_description()
    return (df.withColumn('run_date', psf.lit(run_date))
            .withColumn('git_version', psf.lit(git_version)))


def get_logger(logger_name, logging_path=None):
    """

    Parameters
    ----------
    logger_name: str
        Log name corresponding to the experiment
    logging_path: str
        Path where log file is stored, by default in cwd

    Returns
    -------
    logging.Logger
    """
    logging_path = logging_path if logging_path else os.path.join(os.getcwd(), f'{logger_name}.log')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logging_path, mode='w')
    logger.addHandler(file_handler)
    return logger
