import csv
import requests
import subprocess
import json

all = []
commits = {}

with open(r"C:\Users\ASUS\Desktop\badawczy2\project-kb-main\project-kb-main\MSR2019\notebooks\dataset_msr2019.csv") as f:
    reader = csv.reader(f)
    next(reader)


    for row in reader:

        url_repo = row[1]
        commit_hash = row[2]
        cls = row[3]


        parts = url_repo.split('/')
        owner = parts[-2]
        repo = parts[-1]
        token = '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'

        curl_cmd = f'curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer {token}" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}'
        process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if output:
            data = json.loads(output)

            if 'commit' in data and len(data['commit']) > 0 and 'files' in data and len(data['files']) > 0 :
                commit_message = data['commit']['message']
                commit_id = data['sha']
                files = data['files']
                patches = []
                for file in files:
                    if 'patch' in file:
                        single_patch = file['patch']
                        patches.append(single_patch)

                if cls == "pos":
                    label = 1
                if cls == "neg":
                    label = 0
                
                title = ''
                body = ''
                comments = ''

                
                curl_cmd = f'curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer {token}" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}/pulls'
                process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, shell=True)
                output, error = process.communicate()
                if output:
                    data = json.loads(output)

                    if isinstance(data, list):
                        if len(data) > 0 and 'title' in data[0] and len(data[0]['title']) > 0 and 'body' in data[0] and len(data[0]['body']) > 0:
                            title = data[0]['title']
                            body = data[0]['body']
                            if 'comments_url' in data[0]:
                                comments = data[0]['comments_url']

                                curl_cmd = f'curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer {token}" -H "X-GitHub-Api-Version: 2022-11-28" {comments}'
                                process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, shell=True)
                                output, error = process.communicate()
                                if output:
                                    dataa = json.loads(output)
                                    comment_response = dataa
                                    comments = []
                                    for comment in comment_response:
                                        body_comment = comment['body']
                                        comments.append(body_comment)
                    else:
                            if 'title' in data and len(data['title']) > 0 and 'body' in data and len(data['body']) > 0:
                                title = data['title']
                                body = data['body']
                                if 'comments_url' in data:
                                    comments = data['comments_url']

                                    curl_cmd = f'curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer {token}" -H "X-GitHub-Api-Version: 2022-11-28" {comments}'
                                    process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, shell=True)
                                    output, error = process.communicate()
                                    if output:
                                        dataa = json.loads(output)
                                        comment_response = dataa
                                        comments = []
                                        for comment in comment_response:
                                            body_comment = comment['body']
                                            comments.append(body_comment)

                commits = {
                    'id': commit_id,
                    'message': commit_message,
                    'issue' : {'title': title,
                               'body': body,
                               'comments': comments},
                    'patch': patches,
                    'label': label,
                    'url': url_repo
                }
        
                all.append(commits)
            
        else:
            print(f'Błąd: {error}')
        
with open('commits.json', 'w') as json_file:
    json.dump(all, json_file, indent=2)
