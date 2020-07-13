#!/bin/bash

for dname in `ls source/`; do
    echo $dname
    mv "${dname}/*.pkl" "../../dataset/source/${dname}"
done