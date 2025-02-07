import json
import sng_parser
import threading
from queue import Queue

def worker(sentence_queue, result_list):
    while not sentence_queue.empty():
        sentence = sentence_queue.get()
        try:
            graph = sng_parser.parse(sentence['caption'])
            res = get_relations(graph)
            result_list.append(res)
        finally:
            sentence_queue.task_done()

def get_relations(graph):
    ent = graph['entities']
    res = []
    for re in graph['relations']:
        subject = ent[re['subject']]['head']
        relation = re['relation']
        object = ent[re['object']]['head']
        res.append(f"{subject} {relation} {object}")
    return res

# Load data
with open('YOUR DATA PATH', 'r') as f:
    data = json.load(f)

# Prepare queue and result list
sentence_queue = Queue()
result_list = []

# Fill the queue with sentences
for sentence in data['sentences']:
    sentence_queue.put(sentence)

# Start threads
num_threads = 80  # Adjust based on your machine's capability
threads = []
for _ in range(num_threads):
    thread = threading.Thread(target=worker, args=(sentence_queue, result_list))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Update data dictionary with results
data['scene_graph'] = result_list

with open('RESULT PATH', 'w') as f:
    json.dump(data, f, indent=4) 