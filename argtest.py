import argparse

parser = argparse.ArgumentParser(description="Train classifiers using LLM.")

#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                            help='an integer for the accumulator')

parser.add_argument('-tr', '--train', type=str, required=True, help='Set train data (mandatory)')
parser.add_argument('-te', '--test', type=str, help='Set test data')
parser.add_argument('-d', '--dir', type=str, help='Set a base directory for the train and test data')

parser.add_argument('-m', '--model', type=str, required=True, help='Set the base model for training')
parser.add_argument('-e', '--epoch', type=int, default=5, help='Set number of epochs for the training')
parser.add_argument('-b', '--batch', type=int, default=32, help='Set number of batchs for the training')
parser.add_argument('-c', '--crt', type=str, help='Set the crt file for the certification')
parser.add_argument('-n', '--num_labels', type=int, default=2, help='Set number of labels to classify')
parser.add_argument('--foo', action='store_true')

args = parser.parse_args()

print(args)

if args.dir is None :
    base_dir = '.'
else :
    base_dir = args.dir

if args.test:
    test_data = base_dir + '/' + args.test
else:
    test_data = None

if args.foo:
    print("foo is true")

print(test_data)
