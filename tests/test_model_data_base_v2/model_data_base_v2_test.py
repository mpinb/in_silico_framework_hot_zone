from model_data_base.model_data_base_v2 import ModelDataBase, MdbException
from model_data_base import model_data_base_register
from model_data_base.model_data_base_v2 import get_versions
import model_data_base.IO.LoaderDumper.to_pickle as to_pickle
from model_data_base.IO.LoaderDumper import pandas_to_msgpack
import pytest, os, shutil, six, tempfile, warnings, subprocess
import numpy as np
from getting_started import parent as getting_started_parent
import pandas
from pandas.util.testing import assert_frame_equal
from model_data_base.mdb_initializers.load_simrun_general import init
from model_data_base.utils import silence_stdout
from model_data_base import IO

parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def test_unique_id_is_set_on_initialization(empty_mdb_v2):
    assert empty_mdb_v2._unique_id is not None


def test_unique_id_stays_the_same_on_reload(empty_mdb_v2):
    mdb1 = empty_mdb_v2
    mdb2 = ModelDataBase(empty_mdb_v2.basedir)
    assert mdb1._unique_id == mdb2._unique_id


def test_new_unique_id_is_generated_if_it_is_not_set_yet(empty_mdb_v2):
    empty_mdb_v2._unique_id = None
    empty_mdb_v2.save_db()
    assert empty_mdb_v2._unique_id is None
    mdb = ModelDataBase(empty_mdb_v2.basedir)
    assert mdb._unique_id is not None


def test_get_dumper_string_by_dumper_module(empty_mdb_v2):
    '''dumper string should be the modules name wrt IO.LoaderDumpers'''
    s1 = IO.LoaderDumper.get_dumper_string_by_dumper_module(to_pickle)
    s2 = 'to_pickle'
    assert s1 == s2


def test_get_dumper_string_by_savedir(empty_mdb_v2):
    '''dumper string should be the same if it is determined
    post hoc (by providing the path to an already existing folder)
    or from the module reference directly.'''
    empty_mdb_v2.setitem('test', 1, dumper=to_pickle)
    s1 = empty_mdb_v2._detect_dumper_string_of_existing_key('test')
    s2 = IO.LoaderDumper.get_dumper_string_by_dumper_module(to_pickle)
    assert s1 == s2


def test_can_detect_self_as_dumper(empty_mdb_v2):
    '''dumper string should be the same if it is determined
    post hoc (by providing the path to an already existing folder)
    or from the module reference directly.'''
    empty_mdb_v2.setitem('test', 1, dumper='self')
    s1 = empty_mdb_v2._detect_dumper_string_of_existing_key('test')
    assert s1 == 'self'


def test_metadata_update(empty_mdb_v2):
    '''the method _update_metadata_if_necessary has the purpose of
    providing a smooth transition from databases, that had not implemented
    metadata to the newer version. This function should not overwrite
    existing metadata'''
    empty_mdb_v2.setitem('test', 1, dumper='self')
    empty_mdb_v2.setitem('test2', 1, dumper=to_pickle)
    msg1 = "{} =/= {}".format(empty_mdb_v2['test'].metadata['version'],
                              get_versions()['version'])
    msg2 = "{} =/= {}".format(empty_mdb_v2['test2'].metadata['version'],
                              get_versions()['version'])
    msg_git = "\nDid the commit turn dirty during testing?\n"
    msg_git += subprocess.check_output(['git status'],
                                       shell=True).decode('utf-8')
    assert empty_mdb_v2['test'].metadata['version'] == get_versions(
    )['version'], msg1 + msg_git
    assert empty_mdb_v2['test2'].metadata['version'] == get_versions(
    )['version'], msg2 + msg_git
    assert empty_mdb_v2.['test'].metadata['dumper'], 'self'
    assert empty_mdb_v2.['test2'].metadata['dumper'] == 'to_pickle'
    assert empty_mdb_v2.['test'].metadata[
        'metadata_creation_time'] == 'together_with_new_key'
    assert empty_mdb_v2['test2'].metadata[
        'metadata_creation_time'] == 'together_with_new_key'

    # directly after deleting metadata database, every information is "unknown"
    metadata_db_path = os.path.join(empty_mdb_v2.basedir, 'test2', 'metadata.json')
    assert os.path.exists(metadata_db_path)
    os.remove(metadata_db_path)

    assert empty_mdb_v2['test'].metadata['dumper'] == 'unknown'
    assert empty_mdb_v2['test2'].metadata['dumper'] == 'unknown'

    #after initialization, the metdata is rebuild
    mdb = ModelDataBase(empty_mdb_v2.basedir)
    assert mdb.['test'].metadata['version'], "unknown"
    assert mdb.['test2'].metadata['version'] == "unknown"
    assert mdb.['test'].metadata['dumper'] == 'self'
    assert mdb.['test2'].metadata['dumper'] == 'to_pickle'
    assert mdb.['test'].metadata['metadata_creation_time'] == 'post_hoc'
    assert mdb.['test2'].metadata['metadata_creation_time'] == 'post_hoc'


def test_check_working_dir_clean_for_build_works_correctly():
    #can create database in empty folder
    testpath = tempfile.mkdtemp()

    #cannot create database if file is in folder
    with open(os.path.join(testpath, 'somefile'), 'w'):
        pass
    with pytest.raises(MdbException):
        ModelDataBase(testpath)

    #can create database if folder can be created but does not exist
    shutil.rmtree(testpath)
    ModelDataBase(testpath)

    #cannot create database if subfolder is in folder
    shutil.rmtree(testpath)
    os.makedirs(os.path.join(testpath, 'somefolder'))
    with pytest.raises(Exception):
        ModelDataBase(testpath)

    #tidy up
    shutil.rmtree(testpath)


def test_mdb_does_not_permit_writes_if_readonly(empty_mdb_v2):
    mdb = empty_mdb_v2
    mdb.readonly = True

    def fun():
        mdb['test'] = 1

    with pytest.raises(MdbException):
        fun()


def test_mdb_will_not_be_created_if_nocreate(empty_mdb_v2):
    testpath = tempfile.mkdtemp()
    with pytest.raises(Exception):
        ModelDataBase(testpath, nocreate=True)
    ModelDataBase(empty_mdb_v2.basedir, nocreate=True)
    shutil.rmtree(testpath)


def test_managed_folder_really_exists(empty_mdb_v2):
    empty_mdb_v2.create_managed_folder('asd')
    assert os.path.exists(empty_mdb_v2['asd'])

    #deleting the db entry deletes the folder
    folder_path = empty_mdb_v2['asd']
    del empty_mdb_v2['asd']
    assert not os.path.exists(folder_path)


def test_managed_folder_does_not_overwrite_existing_keys(empty_mdb_v2):
    empty_mdb_v2.create_managed_folder('asd')
    with pytest.raises(MdbException):
        empty_mdb_v2.create_managed_folder('asd')


def test_can_instantiate_sub_mdb(empty_mdb_v2):
    empty_mdb_v2.create_sub_mdb('test_sub_mdb')
    assert isinstance(empty_mdb_v2['test_sub_mdb'], ModelDataBase)


def test_cannot_set_hierarchical_key_it_is_already_used_in_hierarchy(empty_mdb_v2):
    empty_mdb_v2['A'] = 1
    empty_mdb_v2[('B', '1')] = 2

    def fun():
        empty_mdb_v2['A', '1'] = 1

    def fun2():
        empty_mdb_v2['B'] = 1

    def fun3():
        empty_mdb_v2['B', '1'] = 1

    def fun4():
        empty_mdb_v2['B', '1', '2'] = 1

    with pytest.raises(MdbException):
        fun()
    with pytest.raises(MdbException):
        fun2()
    fun3()
    with pytest.raises(MdbException):
        fun4()


def test_keys_of_sub_mdbs_can_be_called_with_a_single_tuple(empty_mdb_v2):
    sub_mdb = empty_mdb_v2.create_sub_mdb(('A', '1'))
    sub_mdb['B'] = 1
    sub_mdb['C', '1'] = 2
    assert empty_mdb_v2['A', '1', 'B'] == 1
    assert empty_mdb_v2['A', '1', 'C', '1'] == 2


def test_sub_mdb_does_not_overwrite_existing_keys(empty_mdb_v2):
    empty_mdb_v2.setitem('asd', 1)
    with pytest.raises(MdbException):
        empty_mdb_v2.create_sub_mdb('asd')


def test_can_set_items_using_different_dumpers(empty_mdb_v2):
    empty_mdb_v2.setitem('test_self', 1, dumper='self')
    empty_mdb_v2.setitem('test_to_pickle', 1, dumper=to_pickle)
    assert empty_mdb_v2['test_self'] == empty_mdb_v2['test_to_pickle']


def test_setitem_allows_replacing_an_existing_key_while_simultaneously_using_it(
        empty_mdb_v2):
    empty_mdb_v2['test'] = 1
    empty_mdb_v2['test'] = empty_mdb_v2['test'] + 1
    assert empty_mdb_v2['test'] == 2


def test_deleting_an_item_really_deletes_it(empty_mdb_v2):
    empty_mdb_v2['test'] = 1
    del empty_mdb_v2['test']
    assert len(list(empty_mdb_v2.keys())) == 0


def test_overwriting_an_item_deletes_old_version(empty_mdb_v2):

    def count_number_of_subfolders(key):
        '''counts the number of folders that are in the mdb
        basedir that match a given key'''
        folders = [f for f in os.listdir(empty_mdb_v2.basedir) if key in f]
        return len(folders)

    assert count_number_of_subfolders('test') == 0
    empty_mdb_v2['test'] = 1
    assert count_number_of_subfolders('test') == 1
    empty_mdb_v2['test'] = 1
    assert count_number_of_subfolders('test') == 1
    del empty_mdb_v2['test']
    assert count_number_of_subfolders('test') == 0


def test_maybe_calculate_runs_calculation_the_first_time_and_gets_result_from_mdb_afterwards(
        empty_mdb_v2):
    mdb = empty_mdb_v2
    flag = []

    def fun():
        flag.append('fun_was_called')
        return 1

    res = empty_mdb_v2.maybe_calculate(
        'my_key_where_result_of_fun_should_be_stored', fun)
    assert res == 1
    assert len(list(empty_mdb_v2.keys())) == 1
    assert len(flag) == 1
    res = empty_mdb_v2.maybe_calculate(
        'my_key_where_result_of_fun_should_be_stored', fun)
    assert res == 1
    assert len(list(empty_mdb_v2.keys())) == 1
    assert len(flag) == 1
    fun()
    assert len(flag) == 2


def test_accessing_non_existent_key_raises_KeyError(empty_mdb_v2):
    with pytest.raises(KeyError):
        empty_mdb_v2['some_nonexistent_key']


@pytest.mark.skipif(
    six.PY3,
    reason=
    "Old mdb only readable in Py2. Py3 does not have pandas.indexes. It has been moved to pandas.core.indexes."
)
def test_compare_old_mdb_with_freshly_initialized_one(client):
    '''ensure compatibility with old versions'''
    old_path = os.path.join(parent, \
                            'test_model_data_base', \
                            'data',\
                            'already_initialized_mdb_for_compatibility_testing')
    old_mdb = ModelDataBase(old_path, readonly=True, nocreate=True)

    # Manually create mdb
    path = tempfile.mkdtemp()
    fresh_mdb_v2 = ModelDataBase(path)
    test_data_folder = os.path.join(getting_started_parent,
                                    'example_simulation_data')
    with silence_stdout:
        init(fresh_mdb_v2,
             test_data_folder,
             client=client,
             rewrite_in_optimized_format=False,
             parameterfiles=False,
             dendritic_voltage_traces=False,
             dumper = pandas_to_msgpack)

    #old_mdb['reduced_model']
    assert_frame_equal(fresh_mdb_v2['voltage_traces'].compute().sort_index(axis=1), \
                        old_mdb['voltage_traces'].compute().sort_index(axis=1))
    assert_frame_equal(fresh_mdb_v2['synapse_activation'].compute().sort_index(axis=1), \
                        old_mdb['synapse_activation'].compute().sort_index(axis=1))
    assert_frame_equal(fresh_mdb_v2['cell_activation'].compute().sort_index(axis=1), \
                        old_mdb['cell_activation'].compute().sort_index(axis=1))
    assert_frame_equal(fresh_mdb_v2['metadata'].sort_index(axis=1), \
                        old_mdb['metadata'].sort_index(axis=1))

    # cleanup
    shutil.rmtree(path)

    # reduced model can be loaded - commented out by Rieke during python 2to3 transition
    #     Rm = old_mdb['reduced_lda_model']
    #     Rm.plot() # to make sure, this can be called


def test_check_if_key_exists_can_handle_str_and_tuple_keys(empty_mdb_v2):
    empty_mdb_v2['a'] = 1
    empty_mdb_v2['b', 'b'] = 1
    assert empty_mdb_v2.check_if_key_exists('a')
    assert empty_mdb_v2.check_if_key_exists(('a',))
    assert empty_mdb_v2.check_if_key_exists(('b', 'b'))
    assert not empty_mdb_v2.check_if_key_exists(('a', 'b'))
    assert not empty_mdb_v2.check_if_key_exists('b')


def test_dumper_can_be_updated_and_metadata_is_adapted(empty_mdb_v2):
    empty_mdb_v2.setitem('a', 1, dumper='self')
    m = empty_mdb_v2.metadata['a']
    assert m['dumper'] == 'self'
    empty_mdb_v2.change_dumper('a', to_pickle)
    m = empty_mdb_v2.metadata['a']
    assert m['dumper'] == 'to_pickle'
    assert 'dumper_update' in list(m.keys())
    du = m['dumper_update']
    assert du[1]['dumper'] == 'to_pickle'