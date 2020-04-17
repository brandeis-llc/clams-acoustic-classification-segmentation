import sys, json, ftfy

def convert(inpath, outpath):
	word_list = []
	with open(inpath) as infile:
		output_dict = json.loads(infile.read(), encoding='utf-8')
		for d in output_dict['words']:
			word = d['word']
			word_list.append(ftfy.fix_text(word).lower())
			
	with open(outpath, 'w') as outfile:
		transcript = ' '.join(word_list)
		outfile.write(transcript)





