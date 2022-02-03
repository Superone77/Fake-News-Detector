"""
Command Line Interface

python cli.py <statement> --subject <subject> --speaker <speaker> --job <job> --state <state> --party <party>
 --venue <venue> --help


"""

from para_list import *
from pre_processing import *
from predict import *
import sys, getopt,warnings
warnings.filterwarnings('ignore')

stmt = None
subject = None
speaker = None
job = None
state = None
party = None
venue = None


try:
   opts, args = getopt.getopt(sys.argv[1:],'hsb:sp:j:st:p:v:',["help","subject=","speaker=","job=","state=","party=","venue="])
except getopt.GetoptError:
   print('python cli.py --help')
   sys.exit(2)
if len(args)>0:
   stmt = str(args[0])
if stmt == None:
   stmt = ""
for opt, arg in opts:
   if opt == "--help":
      help()
      sys.exit()
   elif opt =="--subject":
      subject = arg
   elif opt =="--speaker":
      speaker = arg
   elif opt =="--job":
      job = arg
   elif opt =="--state":
      state = arg
   elif opt =="--party":
      party = arg
   elif opt =="--venue":
      venue = arg

meta = metadata_processing(party,state,venue,job,subject,speaker)
stmt = stmt
word_id, pos_id, dep_id = stmt_processing(stmt)
filepath = "model/lstm"
label_pred, percent_pred = predict(filepath, word_id, pos_id, dep_id, meta)
result = []
result.append(label_pred)
result.append(percent_pred)
print(result)









