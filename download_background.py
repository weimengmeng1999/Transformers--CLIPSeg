import argparse
import os
import synapseclient
import synapseutils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', required=True, help='Your Synapse username.')
    parser.add_argument('--password', required=True, help='Your Synapse password.')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory.')
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Connect to Synapse
    syn = synapseclient.Synapse()
    syn.login(args.username, args.password)

    # Download training and validation dataset
    files = synapseutils.syncFromSynapse(syn, 'syn22438512', args.output_dir)
    print('The data has been successfully downloaded into ' + args.output_dir)

if __name__ == '__main__':
    main()