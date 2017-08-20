import os

def generate_param_files_with_valid_references():
    IN_SILICO_FRAMEWORK_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    suffix = '.TEMPLATE'
    filelist = ['getting_started/biophysical_constraints/86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param.TEMPLATE', \
                'getting_started/functional_constraints/network.param.TEMPLATE']
    for path in filelist:
        path = os.path.join(IN_SILICO_FRAMEWORK_DIR, path)
        print path
        assert(os.path.exists(path))
        assert(path.endswith(suffix))
        with open(path, 'r') as in_, open(path.rstrip(suffix), 'w') as out_:
            for line in in_.readlines():
                line = line.replace('[IN_SILICO_FRAMEWORK_DIR]', IN_SILICO_FRAMEWORK_DIR)
                out_.write(line)
            
generate_param_files_with_valid_references()