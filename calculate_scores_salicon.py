import pysaliency

dataset_location = '../SALICON'
stimuli_salicon_val, fixations_salicon_val = pysaliency.get_SALICON_val(location=dataset_location)
gold_standard = pysaliency.FixationMap(stimuli_salicon_val, fixations_salicon_val, kernel_size=35)
my_model = pysaliency.SaliencyMapModelFromDirectory(stimuli_salicon_val, "../salicon_output")


def scores():
    auc_j = my_model.AUC(stimuli_salicon_val, fixations_salicon_val, nonfixations='uniform')
    print("AUC-j: ", auc_j)

    sim = my_model.SIM(stimuli_salicon_val, gold_standard)
    print("SIM: ", sim)

    s_auc = my_model.AUC(stimuli_salicon_val, fixations_salicon_val, nonfixations='shuffled')
    print("s-AUC: ", s_auc)

    cc = my_model.CC(stimuli_salicon_val, gold_standard)
    print("CC: ", cc)

    nss = my_model.NSSs(stimuli_salicon_val, fixations_salicon_val)
    print("NSS: ", nss.mean())


scores()
