from . import data, net


def predict(filename):
    print('File:')
    print(filename, '\n')

    tags, coordinate_inputs, coordinate_texts, coordinate_tags = data.prepare_data(filename)

    print('Expected category tags:')
    print(tags, '\n')

    # print('Input coordinates:')
    # print(coordinate_inputs)
    print('Expected coordinate tags:')
    print(coordinate_tags)

    predictions = net.evaluate(coordinate_inputs)
    predicted_coordinate_tags = []
    for prediction in predictions:
        predicted_coordinate_tags.append(prediction[1])
    probabilities = []
    for prediction in predictions:
        probabilities.append("{:.0%}".format(prediction[0]))
    print('Predicted coordinate tags:')
    print(predicted_coordinate_tags)
    print('Confidence:')
    print(probabilities, '\n')

    predicted_tags = data.combine_predicted_tags(coordinate_texts, predicted_coordinate_tags)
    print(predicted_tags)
