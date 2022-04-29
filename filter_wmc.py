#!/usr/bin/env python3

import os, sys
import json
import pandas as pd
import numpy as np
import scipy.io as sio
import nibabel as nib

def load_labels(labels_path):

    print('loading filter streamlines assignments labels datatype')

    # labels assignments
    labels = pd.read_csv(labels_path,header=None).rename(columns={0: 'index'})

    return labels

def load_classification_structure(classification_path):

    # load classification
    print('loading classification')
    classification = sio.loadmat(classification_path)

    # extract names and indices from classification
    names = list(np.ravel(list(classification['classification'][0]['names'][0][0])))

    indices = classification['classification'][0]['index'][0][0]

    return classification, names, indices

def filter_wmc(labels,classification,indices):

    print('filtering wmc datatype')

    # identify new indices
    new_indices = indices * np.array(labels["index"].values)

    # set up new classification
    classification['classification'][0]['index'][0][0] = new_indices

    return classification, new_indices

def filter_visualizer_jsons(streamlines,names,new_indices,outdir):

    print('generating filtered json files for visualization')
    tractsfile = []
    dumby_indices = [ f+1 for f in range(len(names)) ]
    for i in range(len(names)):
        idx = i

        stm_idxs = np.where(new_indices == i)[0]

        streams = np.zeros([len(stm_idxs)],dtype=object)
        for e in range(len(streams)):
            streams[e] = np.transpose(streams[stm_idxs][e]).round(2)

        color=list(cm.nipy_spectral(i))[0:3]
        count = len(stm_idxs)

        print("sub-sampling for json")
        if count < 1000:
            max = count
        else:
            max = 1000
        jsonfibers = np.reshape(streams[:max], [max,1]).tolist()
        for j in range(max):
            jsonfibers[j] = [jsonfibers[j][0].tolist()]

        with open ('wmc/tracts/'+str(i)+'.json', 'w') as outfile:
            jsonfile = {'name': names[i-1], 'color': color, 'coords': jsonfibers}
            json.dump(jsonfile, outfile)

        tractsfile.append({"name": names[i-1], "color": color, "filename": str(i+1)+'.json'})

    with open ('wmc/tracts/tracts.json', 'w') as outfile:
        json.dump(tractsfile, outfile, separators=(',', ': '), indent=4)

def main():

    # load config
    with open('config.json','r') as config_f:
        config = json.load(config_f)

    # make output directories
    outdir = 'wmc'

    # load classification
    classification, names, indices = load_classification_structure(config['classification'])

    # labels
    labels = load_labels(config['index'])

    # filter classification structure
    classification, new_indices = filter_wmc(labels,classification,indices)

    # save classification structure
    print("saving classification.mat")
    sio.savemat(outdir+'/classification.mat', { "classification": {"names": names, "index": new_indices }})

    # load tractogram
    print('loading tractogram')
    track = nib.streamlines.load(config['track'])

    # create visualizer jsons
    filter_visualizer_jsons(track.streamlines,names,new_indices)
