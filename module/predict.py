from . import data, net


def predict(filename, model_path=net.SERIALIZED_MODEL_NAME):
    print('File:')
    print(filename, '\n')

    tags, coordinate_inputs, coordinate_texts, coordinate_tags = data.prepare_data(filename)
    coordinate_inputs = [coordinate_inputs[i][:8] for i in range(len(coordinate_inputs))]
    
    print('Expected category tags:')
    print(tags, '\n')

    # print('Input coordinates:')
    # print(coordinate_inputs)
    print(coordinate_inputs)
    print(coordinate_texts)
    print('Expected coordinate tags:')
    print(coordinate_tags)

    predictions = net.evaluate(coordinate_inputs, model_path)
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


def get_expected_tags(filename):
    tags, coordinate_inputs, coordinate_texts, coordinate_tags = data.prepare_data(filename)
    return coordinate_tags


def get_predicted_tags(filename, model_path=net.SERIALIZED_MODEL_NAME):
    tags, coordinate_inputs, coordinate_texts, coordinate_tags = data.prepare_data(filename)
    coordinate_inputs = [coordinate_inputs[i][:8] for i in range(len(coordinate_inputs))]
    predictions = net.evaluate(coordinate_inputs, model_path)
    predicted_coordinate_tags = []
    for prediction in predictions:
        predicted_coordinate_tags.append(prediction[1])
    return predicted_coordinate_tags
