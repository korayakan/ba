def predict(line, n_predictions=3):
    print(line)
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

if __name__ == '__main__':
    predict(sys.argv[1])