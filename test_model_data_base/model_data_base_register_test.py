from .context import *
from model_data_base.model_data_base import ModelDataBase, MdbException
from model_data_base.model_data_base_register import _get_mdb_register, \
        ModelDataBaseRegister, get_mdb_by_unique_id, register_mdb
import unittest

def assert_search_mdb_did_not_fail(mdbr):
    keys = mdbr.mdb.keys()
    keys = [k for k in keys if isinstance(k, tuple)]
    #for k in keys: print (mdbr.mdb[k])
    assert(not keys)
    
class Tests(unittest.TestCase):       
    def setUp(self):        
        self.basetempdir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.basetempdir)     
        
    def test_get_mdb_register_raises_mdb_exception_if_there_is_no_register(self):
        self.assertRaises(MdbException, lambda:  _get_mdb_register(self.basetempdir))
    
    def test_added_mdb_can_be_found_by_id(self):
        p1 = os.path.join(self.basetempdir, 'test1')
        p2 = os.path.join(self.basetempdir, 'test1', 'test2')
        p3 = os.path.join(self.basetempdir, 'test2', 'test2')
        mdb1 = ModelDataBase(p1)
        mdb2 = ModelDataBase(p2)
        mdb3 = ModelDataBase(p3)
        
        mdbr = ModelDataBaseRegister(self.basetempdir)
        
        self.assert_(get_mdb_by_unique_id(self.basetempdir, mdb1.get_id()).basedir == p1)
        self.assert_(get_mdb_by_unique_id(p1, mdb2.get_id()).basedir == p2)
        self.assert_(get_mdb_by_unique_id(p3, mdb3.get_id()).basedir == p3)
        
        mdb4 = ModelDataBase(os.path.join(self.basetempdir, 'test4'))
        register_mdb(mdb4)
        self.assert_(get_mdb_by_unique_id(p2, mdb4.get_id()).basedir == mdb4.basedir)
        assert_search_mdb_did_not_fail(mdbr)
        
    def test_unknown_id_raises_KeyError(self):
        mdbr = ModelDataBaseRegister(self.basetempdir)
                
        self.assertRaises(KeyError, lambda: get_mdb_by_unique_id(self.basetempdir, 'bla'))
        assert_search_mdb_did_not_fail(mdbr)
        
        
# test_search_mdbs_finds_mdbs