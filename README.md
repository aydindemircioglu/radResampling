
# Resampling in radiomics

This repository contains the code for the paper
'The effect of data resampling methods in radiomics'.

The repository consists of Python code (in ./), data (in ./data), results (in ./results) and data for the paper (./data).

To restart the whole experiment and re-generate the results:

```python3 ./experiment.py```

To ensure that all libraries are available, install the requirements.txt by
```pip3 install -r requirements.txt```.

Modify ```./parameters.py``` to your needs, e.g. change number of cpus or
number of repeats. The current configuration will take around 5-6 days on a
32-core CPU (AMD Threadripper 2950X).

After all experiments finished, or if you just want to re-evaluate the
provided results, execute:

```python3 ./evaluate.py```

Note that re-evaluation is possible without running the experiments, since
all the results are
store within this repository.


# License

Datasets are licensed by their respective owners.


## MIT License

Copyright (c) 2023 aydin demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



#
