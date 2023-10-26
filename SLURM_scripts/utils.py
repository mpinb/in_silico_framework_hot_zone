import six
import distributed
import os


def get_user_port_numbers():
    """Read the port numbers defined in config/port_numbers.ini

    Returns:
        dict: port numbers as values, names as keys
    """
    import configparser
    # user-defined port numbers
    __parent_dir__ = os.path.realpath(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(__parent_dir__, "config", "port_numbers.ini"))
    ports = config['PORT_NUMBERS']
    return ports


def get_client():
    """Gets the distributed.client object if dask has been setup.
    Assumes the dask client is succesfully running on the port numbers defined in config/port_numbers.ini
    This assumption can be False is:
    - The ports were already in use, and Dask jumped to the next available port number.
    - The dask client was not set up at all, for whatever reason (e.g. connectivity problems)

    Returns:
        Client: the client object
    """
    ports = get_user_port_numbers()
    if six.PY2:
        client_port = ports['dask_client_2']
    else:
        client_port = ports['dask_client_3']

    from socket import gethostbyname, gethostname
    hostname = gethostname()
    ip = gethostbyname(hostname)
    if 'soma' in hostname:  # [somalogin0*, somagpu*, somacpu*]
        #we're on the soma cluster and have infiniband
        ip_infiniband = ip.replace('100', '102')  # a bit hackish, but it works
        client = distributed.Client(ip_infiniband + ':' + client_port)
    else:
        client = distributed.Client(ip + ':' + client_port)
    return client

def tear_down_cluster(client):
    import os
    def get_jobid():
        return os.environ['SLURM_JOBID']
    str_ = 'scancel '
    for jid in set(client.run(get_jobid).values()):
        str_ += jid + ' '
    print(str_)