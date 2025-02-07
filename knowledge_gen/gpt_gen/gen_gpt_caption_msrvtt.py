import openai
import json
from threading import Thread
from queue import Queue
import time

openai.api_key = 'YOUR API KEY'  
openai.api_base = "YOUR API BASE"

def call_chatgpt(prompt, output_queue, order):
    max_retries = 100
    for attempt in range(max_retries):
        try:
            result_txt =""
            for i in range(12):
                prompt.replace("[NEXT]", result_txt)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    n=1,
                )
                print(order)
                txt = response.choices[0].message['content'].strip()
                result_txt += txt
            output_queue.put((order, result_txt))
            return
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:  
                time.sleep(1) 
            else:
                output_queue.put((order, 'wrong'))
                return
    print("error prompt:", prompt)


def thread_worker(input_queue, output_queue):
    while True:
        order, prompt = input_queue.get()
        if prompt is None:
            break  
        call_chatgpt(prompt, output_queue, order)
        input_queue.task_done()

def main(debug=True):
    with open('YOUR DATA PATH JSON', 'r') as f:
        data = json.load(f)
    
    input_queue = Queue()
    output_queue = Queue()
    num_threads = 60  

    threads = []
    for i in range(num_threads):
        thread = Thread(target=thread_worker, args=(input_queue, output_queue))
        thread.start()
        threads.append(thread)

    prompt_template = '''
    Q: This is a subtitle for a video {} . I will take 12 pictures from the video. The previous frame is {}. Please provide possible descriptive sentences for these next pictures:
    A: 
    '''
    for index, sentence in enumerate(data['sentences']):
        if debug:
            if index >10:
                break
        input_prompt = prompt_template.format(sentence['caption'], "[NEXT]")
        input_queue.put((index, input_prompt))
    
    input_queue.join()

    for _ in threads:
        input_queue.put((None, None))
    
    for thread in threads:
        thread.join()

    results = [None] * len(data['sentences'])
    while not output_queue.empty():
        order, text = output_queue.get()
        results[order] = text

    data['gpt_caption'] = results
    with open('RESULT JSON PATH', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main(False)
