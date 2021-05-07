#!/usr/bin/python

#######################################
# prepare .hoc registered to standard #
# barrel cortex for import into       #
# NEURON: apical/dend/axon names are  #
# converted using the NEURON Import3D #
# naming convention (i.e. using       #
# arrays apical[n], dend[m], axon[l]) #
#                                     #
# Written by Robert Egger             #
#######################################

import sys

def read_hoc_file(hoc_filename):
	soma_name_map = {}
	axon_name_map = {}
	basal_name_map = {}
	apical_name_map = {}
	soma_name_count = 0
	axon_name_count = 0
	basal_name_count = 0
	apical_name_count = 0
	hoc_file = open(hoc_filename, 'r')
	for line in hoc_file:
		line = line.replace('BasalDendrite', 'dend')
		if line and 'create' in line:
			if 'soma' in line:
				splitLine = line.strip().split(' ')
				for splitStr in splitLine:
					if 'soma' in splitStr:
						endIndex = max(splitStr.find('}'), splitStr.find('('))
						oldName = splitStr[:endIndex]
						if oldName in soma_name_map:
							break
						newName = 'soma'
						soma_name_map[oldName] = newName
						soma_name_count += 1
						print('old name: {} new name: {}'.format(oldName, newName))
						break
			if 'axon' in line:
				splitLine = line.strip().split(' ')
				for splitStr in splitLine:
					if 'axon' in splitStr:
						endIndex = max(splitStr.find('}'), splitStr.find('('))
						oldName = splitStr[:endIndex]
						if oldName in axon_name_map:
							break
						newName = 'axon[%d]' % axon_name_count
						axon_name_map[oldName] = newName
						axon_name_count += 1
						print('old name: {} new name: {}'.format(oldName, newName))
						break
			if 'dend' in line:
				splitLine = line.strip().split(' ')
				for splitStr in splitLine:
					if 'dend' in splitStr:
						endIndex = max(splitStr.find('}'), splitStr.find('('))
						oldName = splitStr[:endIndex]
						if oldName in basal_name_map:
							break
						newName = 'dend[%d]' % basal_name_count
						basal_name_map[oldName] = newName
						basal_name_count += 1
						print('old name: {} new name: {}'.format(oldName, newName))
						break
			if 'apical' in line:
				splitLine = line.strip().split(' ')
				for splitStr in splitLine:
					if 'apical' in splitStr:
						endIndex = max(splitStr.find('}'), splitStr.find('('))
						oldName = splitStr[:endIndex]
						if oldName in apical_name_map:
							break
						newName = 'apic[%d]' % apical_name_count
						apical_name_map[oldName] = newName
						apical_name_count += 1
						print('old name: {} new name: {}'.format(oldName, newName))
						break
		if line and 'EOF' in line:
			hoc_file.close()
	hoc_file.close()
	structureCounts = {}
	structureCounts['soma'] = soma_name_count
	structureCounts['axon'] = axon_name_count
	structureCounts['dend'] = basal_name_count
	structureCounts['apical'] = apical_name_count
	return structureCounts, soma_name_map, axon_name_map, basal_name_map, apical_name_map

def write_hoc_file(hoc_filename, structureCounts, soma_name_map, axon_name_map, basal_name_map, apical_name_map):
	out_filename = hoc_filename[:-4]
	out_filename += '_NEURON_Import3D_names.hoc'
	hoc_file = open(hoc_filename, 'r')
	out_file = open(out_filename, 'w')
	out_file.write('/***************************************/\n')
	out_file.write('/* Morphology registered to standard   */\n')
	out_file.write('/* barrel cortex and ready for import  */\n')
	out_file.write('/* into NEURON:                        */\n')
	out_file.write('/* apical/dend/axon names are          */\n')
	out_file.write('/* converted using the NEURON Import3D */\n')
	out_file.write('/* naming convention (i.e. using       */\n')
	out_file.write('/* arrays apic[n], dend[m], axon[l])   */\n')
	out_file.write('/*                                     */\n')
	out_file.write('/* Converter written by Robert Egger   */\n')
	out_file.write('/***************************************/\n')
	out_file.write('\n')
	createSomaStr = '{create soma}\n'
	out_file.write(createSomaStr)
	createAxonStr = '{create axon'
	if structureCounts['axon']:
		createAxonStr += '[%d]' % structureCounts['axon']
	createAxonStr += '}\n'
	out_file.write(createAxonStr)
	createBasalStr = '{create dend'
	if structureCounts['dend']:
		createBasalStr += '[%d]' % structureCounts['dend']
	createBasalStr += '}\n'
	out_file.write(createBasalStr)
	createApicalStr = '{create apic'
	if structureCounts['apical']:
		createApicalStr += '[%d]' % structureCounts['apical']
	createApicalStr += '}\n'
	out_file.write(createApicalStr)
	out_file.write('\n')
	for line in hoc_file:
		line = line.replace('BasalDendrite', 'dend')	
		if line and 'create' in line:
			continue
		if line and ('soma' in line or 'axon' in line or 'dend' in line or 'apical' in line):
			splitLine = line.split(' ')
			lineParts = []
			for splitStr in splitLine:
				if 'soma' in splitStr:
					index1 = splitStr.find('}')
					index2 = splitStr.find('(')
					if index1 > -1 and index2 > -1:
						endIndex = min(index1, index2)
					elif index1 == -1 and index2 == -1:
						endIndex = None
					else:
						endIndex = max(index1, index2)
					if endIndex is None:
						oldName = splitStr
					else:
						oldName = splitStr[:endIndex]
					newName = soma_name_map[oldName]
					newName += splitStr[endIndex:]
					lineParts.append(newName)
				elif 'axon' in splitStr:
					index1 = splitStr.find('}')
					index2 = splitStr.find('(')
					if index1 > -1 and index2 > -1:
						endIndex = min(index1, index2)
					elif index1 == -1 and index2 == -1:
						endIndex = None
					else:
						endIndex = max(index1, index2)
					if endIndex is None:
						oldName = splitStr
					else:
						oldName = splitStr[:endIndex]
					newName = axon_name_map[oldName]
					newName += splitStr[endIndex:]
					lineParts.append(newName)
				elif 'dend' in splitStr:
					index1 = splitStr.find('}')
					index2 = splitStr.find('(')
					if index1 > -1 and index2 > -1:
						endIndex = min(index1, index2)
					elif index1 == -1 and index2 == -1:
						endIndex = None
					else:
						endIndex = max(index1, index2)
					if endIndex is None:
						oldName = splitStr
					else:
						oldName = splitStr[:endIndex]
					newName = basal_name_map[oldName]
					newName += splitStr[endIndex:]
					lineParts.append(newName)
				elif 'apical' in splitStr:
					index1 = splitStr.find('}')
					index2 = splitStr.find('(')
					if index1 > -1 and index2 > -1:
						endIndex = min(index1, index2)
					elif index1 == -1 and index2 == -1:
						endIndex = None
					else:
						endIndex = max(index1, index2)
					if endIndex is None:
						oldName = splitStr
					else:
						oldName = splitStr[:endIndex]
					newName = apical_name_map[oldName]
					newName += splitStr[endIndex:]
					lineParts.append(newName)
				else:
					lineParts.append(splitStr)
			newLine = ' '.join(lineParts)
			out_file.write(newLine)
			continue
		if line and 'EOF' in line:
			break
		out_file.write(line)
	hoc_file.close()
	out_file.close()

if __name__ == '__main__':
	hoc_file = ''
	if len(sys.argv) > 1:
		hoc_file = sys.argv[1]
	else:
		hoc_file = six.moves.input('Enter .hoc filename:')
	structureCounts, soma_name_map, axon_name_map, basal_name_map, apical_name_map = read_hoc_file(hoc_file)
	write_hoc_file(hoc_file, structureCounts, soma_name_map, axon_name_map, basal_name_map, apical_name_map)
