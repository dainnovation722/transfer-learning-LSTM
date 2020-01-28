import os
import subprocess


def word_count(path):
    return int(subprocess.check_output(f'detex {path} | wc -m', shell=True))


print('--------------------')
sections = [
    'intro', 'method', 'results', 'conclusion']
total = 0
for sec in sections:
    path = os.path.abspath(f'./src/{sec}/{sec}.tex')
    cnt = word_count(path)
    total += cnt
    print('{:<15}{:>5}'.format(sec, str(cnt)))

print('--------------------')
print('{:<15}{:>5}'.format('main', str(total)))
print('--------------------')
print(f'{"total":<15}{word_count("main.tex"):>5}')
print('--------------------')
