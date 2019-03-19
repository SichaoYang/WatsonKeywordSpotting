"""IBM Watson Transcription Discriminator

This module uses [IBM Watson Speech to Text service](https://www.ibm.com/watson/services/speech-to-text/)
to transcribe audio files. Keyword spotting is performed to give preference to words from a given list.

Accuracy:
    Keyword spotting works for ~90% results. When a keyword is spotted, accuracy >98%.
    Accuracy can be further improved if a separate list of keywords can be provided for each stimulus.
    Otherwise, the best practice is to concatenate audio files, play at 2x speed, and manually check the transcription.

Weakness:
    Keyword spotting does not work when a keyword is a part of the word or phrase actually said,
    e.g. boat-sailboat or coat-suit coat.
    Keyword spotting also ignores disfluencies.

TODOs:
    (1) Disfluencies: compare the timestamps of spotted keywords and the audio onset.
    (2) Word in word: transcribe both with and without keywords and compare the results.
    (3) Confidence: make use of the word confidence returned.
    (4) Long term goal: not only transcribe but also mark out which results can be trusted and which cannot.

.. _IBM Watson Speech to Text service:
   https://www.ibm.com/watson/services/speech-to-text/

"""


import csv
from os.path import join
from watson_developer_cloud import SpeechToTextV1

# (1) Create an IBM Cloud Account at https://cloud.ibm.com/registration.
# (2) Create a Speech to Text service at https://cloud.ibm.com/catalog/services/speech-to-text.
# (*) For convenience, LCNL members can directly contact Sichao Yang for a key to a Lite Speech to Text service.
service = SpeechToTextV1(iam_apikey='YOUR-API-KEY')
# Use the en-US model for American English transcription.
model = service.get_model('en-US_BroadbandModel').get_result()


def get_keywords(index_csv_path):
    """
    Get a {stimulus: [keywords]} dictionary from a csv file, e.g. wdata.csv.
    :param index_csv_path: a csv file containing stimuli-words pairs.
    :return: a {stimulus: [keywords]} dictionary.
    """
    keywords = {}
    with open(index_csv_path, 'r') as index_csv_file:
        csv_reader = csv.DictReader(index_csv_file)
        for row in csv_reader:
            if row['img'] not in keywords:
                if row['secondary']:  # if this row contains dominant and secondary words
                    keywords[row['img']] = [row['dominant'], row['secondary']]
                else:
                    same = []  # try to find an entry in the dictionary with a word in the filename of the current img
                    for img in keywords:
                        if keywords[img][0] in row['img'] or keywords[img][1] in row['img']:
                            same = keywords[img]
                            break
                    if same:
                        keywords[row['img']] = [same[0], same[1]]

    print(keywords)
    return keywords


def transcribe(audio_path, keywords):
    """
    Transcribe an audio file with preference for a list of keywords.
    :param str audio_path: the path of the audio file.
    :param list[str] keywords: a list of keywords to spot in the audio.
    :return: transcription.
    """
    with open(audio_path, 'rb') as audio_file:
        # https://github.com/watson-developer-cloud/python-sdk/blob/master/watson_developer_cloud/speech_to_text_v1.py
        results = service.recognize(
                audio=audio_file,
                content_type='audio/wav',
                timestamps=True,
                word_confidence=True,
                keywords=keywords,
                keywords_threshold=0).get_result()['results']
        # returned DetailedResponse example:
        # {'results':
        #   [{
        #     'keywords_result':
        #       {'wolf':
        #           [{
        #               'normalized_text': 'wolf',
        #               'start_time': 2.0,
        #               'confidence': 0.03,
        #               'end_time': 2.11
        #           }]
        #       },
        #     'alternatives':
        #       [{
        #           'timestamps': [['well', 1.85, 2.11]],
        #           'confidence': 0.57,
        #           'transcript': 'well ',
        #           'word_confidence': [['well', 0.57]]
        #       }],
        #     'final': True
        #   }],
        #   'result_index': 0
        # }
        # When keywords are not spotted, the alternative results cannot be reliable either. Discard them.
        if results and 'keywords_result' in results[0] and results[0]['keywords_result']:
            key = list(results[0]['keywords_result'].keys())[0]  # the spotted keyword
            return results[0]["keywords_result"][key][0]  # word, start and end time, and confidence
        return {'normalized_text': '', 'start_time': '', 'confidence': '', 'end_time': ''}


if __name__ == '__main__':
    data_path = '/home/sichao/Documents/wc/Data'  # change this
    index_csv_path = '/home/sichao/Documents/wc/wdata_combined.csv'
    output_path = '/home/sichao/Documents/wc/output.csv'

    keywords = get_keywords(index_csv_path)

    with open(index_csv_path, 'r') as index_csv_file, open(output_path, 'a+') as output_file:
        csv_reader = csv.DictReader(index_csv_file)
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(csv_reader.fieldnames + ['kws_word', 'kws_start_t', 'kws_conf', 'kws_end_t'])

        line = 1
        for row in csv_reader:
            line += 1  # current line number
            # if line < 1:  # skip some lines before the rows of interest
            #     continue
            # if line > 1000:  # skip some lines after the rows of interest
            #     break

            # C:\Users\LCNL-279-1\Desktop\exp_secondary/Data/rep1*/trgt_audio/rep1*.wav
            path1, path2, path3 = row['audio_path'].split('/')[-3:]
            audio_path = join(data_path, path1, path2, path3)
            print(audio_path)  # print the current audio file to show the progress of the program
            transcript = transcribe(audio_path, keywords[row['img']])
            # write original columns and ['kws_word', 'kws_start_t', 'kws_conf', 'kws_end_t']
            csv_writer.writerow(list(row.values()) + list(transcript.values()))
