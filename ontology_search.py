import json
import os
import csv

def intersection(lst1, lst2):
	return list(set(lst1) & set(lst2))

def search(DESIRED_LABELS, BANNED_LABELS, SEG_CSV):
	"""
	This function searches the Audioset ontology and returns a list
	of youtube ids that have the labels specified by the user and 
	do not have any labels that were banned.

	:param DESIRED_LABELS: English labels you want clips for
		   BANNED_LABELS: English labels you do not want clips for
		   SEG_CSV: Slightly modified audioset csv
	:return 
	"""
	json_path = os.path.join("ontology", "ontology.json")
	with open(json_path, "r") as json_file:
	    json_dump = json.load(json_file)

	desired_ids = []
	banned_ids = []
	all_ids = []
	for obj in json_dump:
		if obj['name'] in DESIRED_LABELS:
			desired_ids.append(obj['id'])
			desired_ids.extend(obj['child_ids'])
		elif obj['name'] in BANNED_LABELS:
			banned_ids.append(obj['id'])
			banned_ids.extend(obj['child_ids'])
		all_ids.append(obj['name'])

	ytids = []
	with open(SEG_CSV, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			pos_labels = row[None]
			# ytid is desired
			if (intersection(pos_labels, desired_ids)) and \
					(not intersection(pos_labels, banned_ids)):
				ytids.append(row['YTID'])

	return ytids

if __name__ == '__main__':
	# DESIRED_LABELS = ["Wind",
	# 				  "Thunderstorm",
	# 				  "Fire",
	# 				  "Inside, small room",
	# 				  "Inside, large room or hall",
	# 				  "Inside, public space",
	# 				  "Reverberation",
	# 				  "Human voice",
	# 				  "Human locomotion",
	# 				  "Hands"]
	DESIRED_LABELS = ["Wind"]
	BANNED_LABELS = ["Bird"]
	SEG_CSV = "eval_segments.csv"

	ytids = search(DESIRED_LABELS, BANNED_LABELS, SEG_CSV)

	for i in ytids: print(i)