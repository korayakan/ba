from data import *
from model import *
import sys

def old_predict(filename, n_predictions=3):
    #output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    #topv, topi = output.data.topk(n_predictions, 1, True)
    #predictions = []

    #for i in range(n_predictions):
    #    value = topv[0][i]
    #    category_index = topi[0][i]
    #    print('(%.2f) %s' % (value, all_categories[category_index]))
    #    predictions.append([value, all_categories[category_index]])

    #return predictions
    print('')


def predict(filename):
    print('File:')
    print(filename, '\n')

    tags, coordinate_inputs, coordinate_texts, coordinate_tags = prepare_data(filename)

    print('Expected tags:')
    print(tags, '\n')

    print(coordinate_inputs)
    print(coordinate_tags, '\n')

    evaluate(coordinate_inputs)

    print('Predicted tags:')
    print('TODO', '\n')


if __name__ == '__main__':
    predict(sys.argv[1])
