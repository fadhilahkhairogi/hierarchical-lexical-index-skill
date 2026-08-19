import os
import sys
import argparse
import time

# Ensure local skill package scripts are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPTS = os.path.join(CURRENT_DIR, 'hierarchical-index', 'scripts')
if os.path.exists(SKILL_SCRIPTS) and SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)

from hierarchical_index import HierarchicalIndex

def main():
    parser = argparse.ArgumentParser(description='Pre-build Hierarchical Index ChromaDB cache for tabular/data lake datasets.')
    parser.add_argument('--dataset-dir', type=str, default=os.getenv('DATALAKE_DIR', './data'),
                        help='Path to dataset directory containing CSV, Excel, TXT, or JSON files.')
    parser.add_argument('--chroma-dir', type=str, default=None,
                        help='Custom directory path where ChromaDB vector database will be stored.')
    parser.add_argument('--workers', type=int, default=50,
                        help='Number of parallel embedding workers (default: 50).')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if args.chroma_dir:
        chroma_dir = os.path.abspath(args.chroma_dir)
    else:
        dataset_parent = os.path.dirname(dataset_dir)
        chroma_dir = os.path.join(dataset_parent, 'hierarchical_index_cache_chroma_db')

    cache_file = os.path.join(os.path.dirname(chroma_dir), 'hierarchical_index_cache.json')

    print('=' * 65)
    print('Hierarchical Index Builder (ChromaDB + BM25 Engine)')
    print(f'Dataset Directory : {dataset_dir}')
    print(f'ChromaDB Directory: {chroma_dir}')
    print(f'Workers           : {args.workers}')
    print('=' * 65)

    if not os.path.exists(dataset_dir):
        print(f'[ERROR] Dataset directory not found: {dataset_dir}')
        print('Please set a valid path via --dataset-dir or DATALAKE_DIR environment variable.')
        sys.exit(1)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    print(f'Building index from: {dataset_dir}')
    print('Connecting to embedding service...')

    t0 = time.time()
    try:
        idx = HierarchicalIndex(dataset_dir=dataset_dir, cache_file=cache_file, max_workers=args.workers)
        print(f'[SUCCESS] Built {len(idx.nodes)} nodes in {time.time() - t0:.2f} seconds.')
        print(f'Cache saved -> {chroma_dir}')
    except Exception as e:
        print(f'[ERROR] Failed to build index: {e}')
        sys.exit(1)

    print('\n[Done] Hierarchical index build complete.')

if __name__ == '__main__':
    main()
