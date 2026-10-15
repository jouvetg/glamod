#!/bin/bash

for A in ??_notebook
do
   zip -r ${A}/${A}.zip ${A} 
done
