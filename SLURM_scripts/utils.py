import six
import os
import logging
logger = logging.getLogger('ISF').getChild(__name__)


def get_user_port_numbers():
    """Read the port numbers defined in config/user_settings.ini

    Returns:
        dict: port numbers as values, names as keys
    """
    import configparser
    # user-defined port numbers
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config = configparser.ConfigParser()
    p = os.path.join(root_dir, "config", "user_settings.ini")
    logger.info("Reading user_settings.ini from {}".format(p))
    config.read(p)
    ports = config['PORT_NUMBERS']
    return ports


def tear_down_cluster(client):
    import os
    def get_jobid():
        return os.environ['SLURM_JOBID']
    str_ = 'scancel '
    for jid in set(client.run(get_jobid).values()):
        str_ += jid + ' '
    print(str_)