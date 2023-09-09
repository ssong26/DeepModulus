"""
Name: Siyuan Song
Date: 2021-02-01

"""

import numpy as np
from odbAccess import openOdb

" Part 0 : Initialize all the position for store"

dic_node = {} # node: (x,y,z) initial position
dic_connectivity = {} # ele_index: (node_list,node_type)



" Part 1 : Read the displacement at given frame"
#

#odb_name = "NH-PE-NoHole.odb"
#odb_name = "NH-PE-Hole.odb"
odb_name = "AB_PS_Hole_Four.odb"
#odb_name = "tilt-9-job.odb"
#
odb = openOdb(odb_name)
#
step = odb.steps[odb.steps.keys()[0]]
#
frame_list = step.frames
#
inst = odb.rootAssembly.instances
subject = inst[inst.keys()[0]]
for v in subject.nodes:
    dic_node[v.label] = (v.coordinates[0],v.coordinates[1])
for v in subject.elements:
    dic_connectivity[v.label] = (v.connectivity,v.type)
#
np.save("dic_node.npy",dic_node)
np.save("dic_connectivity.npy",dic_connectivity)
#
for i,frame in enumerate(frame_list):
    dic_displacement = {} # node: (x,y,z) displacement
    #
    inst_displacement = frame.fieldOutputs['U'].getSubset(region = inst[inst.keys()[0]])
    # Get displacement dictionary
    for v in inst_displacement.values:
        dic_displacement[v.nodeLabel] = (v.data[0],v.data[1])
    #dic_stress = {}
    #stress = frame_list[-1].fieldOutputs['S']
    #inst_stress = stress.getSubset(region = inst[inst.keys()[0]])
    #for v in inst_stress.values:
    #    dic_stress[v.elementLabel] = (v.data[0],v.data[1],v.data[2],v.data[3])
    np.save(str(i) + "_dic_displacement.npy",dic_displacement)
    #np.save("dic_stress.npy",dic_stress)
odb.close()
#
