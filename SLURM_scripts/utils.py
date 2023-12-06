import six
import os


def get_user_port_numbers():
    """Read the port numbers defined in config/user_settings.ini

    Returns:
        dict: port numbers as values, names as keys
    """
    import configparser
    # user-defined port numbers
    __parent_dir__ = os.path.realpath(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(__parent_dir__, "config", "user_settings.ini"))
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