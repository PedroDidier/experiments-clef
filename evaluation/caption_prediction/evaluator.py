import csv
import string
import warnings
import numpy as np
import re
from evaluate import load
from bert_score import BERTScorer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compute caption evaluation metrics on submission for ImageCLEF 2025.')
    parser.add_argument("-i", "--input", help="Input database file path. It needs to be a csv file with IDs and captions.", required=True, dest='input')
    parser.add_argument("-t", "--target", help="Target dataset. It also needs be a csv file with the same IDs and ground-truth captions.", required=True, dest='target')
    parser.add_argument("-o", "--output", help="Output path to save the predictions and scores.", dest='output', required=False)
    args = parser.parse_args()
    return args


class CaptionEvaluator:
    def __init__(self, ground_truth_path, output_path=None, **kwargs):
        """
        This is the evaluator class which will be used for the evaluation.
        Please note that the class name should be `CaptionEvaluator`
        `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
        """
        self.ground_truth_path = ground_truth_path
        self.gt = self.load_gt()
        self.output_path = output_path

        ######## Load Metrics from HuggingFace ########
        print('Loading models from HuggingFace')
        self.scorers = {
            'ROUGE': (self.compute_rouge, load('rouge')),
            # 'bert_scorer': (self.compute_bertscore,
            #     BERTScorer(
            #         model_type="microsoft/deberta-xlarge-mnli",
            #         idf=True
            #     ),
            # ),
            'BLEURT': (self.compute_bleurt, load("bleurt", module_type="metric", checkpoint="bleurt-20"))
        }

    def eval(self, client_payload, _context={}):
        """
        This is the only method that will be called by the framework
        returns a _result_object that can contain up to 2 different scores
        `client_payload["submission_file_path"]` will hold the path of the submission file
        """
        print("Evaluate...")
        # Load submission file path
        submission_file_path = client_payload["submission_file_path"]
        # Load preditctions and validate format
        predictions = self.load_predictions(submission_file_path)

        _result_object = {}

        rows = None
        for i, (score, (score_func, scorer)) in enumerate(self.scorers.items()):
            if i == 0:
                rows = self.compute_score(score_func, scorer, predictions)
            else:
                aux_rows = self.compute_score(score_func, scorer, predictions)
                for row, aux_row in zip(rows, aux_rows):
                    row.append(aux_row[-1])
            _result_object[score] = np.mean([x[-1] for x in rows])

        if self.output_path is not None:
            # Write rows to CSV
            with open(self.output_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write the header row
                writer.writerow(['ID', 'Prediction', 'Caption'] + list(self.scorers.keys()))
                
                # Write the data rows
                writer.writerows(rows)
            
            print(f"Results saved to {self.output_path}")


        return _result_object

    def load_gt(self):
        """
        Load and return groundtruth data
        """
        print("Loading ground truth...")

        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile)

            # Check if the first line is a header and skip it
            first_line = next(reader)

            if "ID" in first_line[0]:
                pass
            else:
                pairs[first_line[0]] = first_line[1]

            for row in reader:
                pairs[row[0]] = row[1]

        return pairs

    def load_predictions(self, submission_file_path):
        """
        Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
        Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
        """
        print("Load predictions...")

        pairs = {}
        image_ids_gt = set(self.gt.keys())
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)

            lineCnt = 0
            occured_images = []
            for row in reader:
                if "ID" not in row[0]:
                    lineCnt += 1

                    # less than two pipe separated tokens on line => Error
                    if(len(row) < 2):
                        self.raise_exception("Wrong format: Each line must consist of an image ID followed by a ',' (comma) and a caption ({}).",
                                                lineCnt, "<imageID><comma><caption>")

                    image_id = row[0]

                    # Image ID does not exist in testset => Error
                    if image_id not in image_ids_gt:
                        self.raise_exception(
                            "Image ID '{}' in submission file does not exist in testset.", lineCnt, image_id)

                    # image id occured at least twice in file => Error
                    if image_id in occured_images:
                        self.raise_exception(
                            "Image ID '{}' was specified more than once in submission file.", lineCnt, image_id)

                    occured_images.append(image_id)

                    pairs[row[0]] = row[1]

        # In case not all images from the testset are contained in the file => Error
        if(len(occured_images) != len(image_ids_gt)):
            self.raise_exception(
                "Number of image IDs in submission file not equal to number of image IDs in testset.", lineCnt)

        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(message.format(
            *args)+" Error occured at record line {}.".format(record_count))

    def pre_process_text(self, text):
        """
        Note: For the relevance metrics (BERT-Score, ROUGE and BLEURT), each caption is pre-processed in the same way:

        - The caption is converted to lower-case.
        - Replace numbers with the token 'number'.
        - Remove punctuation.
        
        Note that the captions are always considered as a single sentence, even if it actually contains several sentences.
        """
        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r'\d+')

        # To lowercase
        text = text.lower()

        # replace numbers with the token 'number'
        text = number_regex.sub('number', text)

        # Remove punctuation using the translator
        text = text.translate(translator)

        return text

    def compute_score(self, score_func, scorer, candidate_pairs):
        rows = []

        for image_key in candidate_pairs:
            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Pre-process text
            candidate_caption = self.pre_process_text(candidate_caption)
            gt_caption = self.pre_process_text(gt_caption)

            # Calculate generic score
            score = score_func(scorer, candidate_caption, gt_caption)[0]
            rows.append([image_key, candidate_caption, gt_caption, score])
        return rows
    
    def compute_bertscore(self, scorer, candidate_caption, gt_caption):
        """
        BERT-Score (Recall) with inverse document frequency (idf) scores computed from the test corpus for importance weighting
        """
        if len(gt_caption) == 0 and len(candidate_caption) == 0:
            bert_score = 1
        # Calculate the BERTScore
        else:
            P, bert_score, F1 = scorer.score([candidate_caption], [gt_caption])

        return bert_score

    def compute_rouge(self, scorer, candidate_caption, gt_caption):
        """
        Recall-Oriented Understudy for Gisting Evaluation (ROUGE) for overlap of unigrams (ROUGE-1) (F-measure)
        """
        if len(gt_caption) == 0 and len(candidate_caption) == 0:
            rouge1_score_f1 = 1
        # Calculate the ROUGE score
        else:
            rouge1_score_f1 = scorer.compute(predictions=[candidate_caption], references=[gt_caption], use_aggregator=False, use_stemmer=False)            

        return rouge1_score_f1["rouge1"]
    
    def compute_bleurt(self, scorer, candidate_caption, gt_caption):
        """
        Bilingual Evaluation Understudy with Representations from Transformers (BLEURT)
        """        
        if len(gt_caption) == 0 and len(candidate_caption) == 0:
            bleurt_score = 1
        # Calculate the BLEURT score
        else:
            bleurt_score = scorer.compute(predictions=[candidate_caption], references=[gt_caption])
        return bleurt_score["scores"]


if __name__ == "__main__":
    args = parse_args()

    ground_truth_path = args.target

    submission_file_path = args.input

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path

    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator
    caption_evaluator = CaptionEvaluator(ground_truth_path, output_path=args.output)

    # Evaluate
    result = caption_evaluator.eval(_client_payload, _context)
    print('Results:', result)
