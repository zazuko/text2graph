import itertools
import json
import urllib
from string import punctuation

import nltk
import spacy
from flask import Flask, request

import neuralcoref
import opennre