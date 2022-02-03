import PySimpleGUIWeb as sg
from para_list import *
from pre_processing import *
from predict import predict

sg.change_look_and_feel("Default")

elements = [[sg.Text(size=[20, 2])],
    [sg.Text("Subject", size=[7, 1]), sg.InputCombo(subjects_list, size=(18, 1)), sg.Text("Speaker", size=[7, 1]), sg.InputCombo(speakers_list, size=(18, 1))],
     [sg.Text("Job", size=[7, 1]), sg.InputCombo(jobs_list, size=(18, 1)),sg.Text("State", size=[7, 1]), sg.InputCombo(states_list, size=(18, 1))], \
         [sg.Text("Party", size=[7, 1]), sg.InputCombo(parties_list, size=(18, 1)),sg.Text("Venue", size=[7, 1]), sg.InputCombo(venues_list, size=(18, 1))],
    ]

statement = [[sg.Text("Fake News Detector", size=[20, 2])],[sg.Multiline('Enter your text', size=(53,3))],[sg.Button("Check"),sg.Button("Exit"),sg.Button("Feedback")]]

layout = [[sg.Column(statement),sg.Column(elements),sg.Text(size = (33,3))]]
window = sg.Window('Fake News Detector', layout, size=(400, 400), resizable=True)
window.finalize()
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    if(event == "Check"):
        print("Checking")
        stmt = values[0]
        subject = values[1]
        speaker = values[2]
        job = values[3]
        state = values[4]
        party = values[5]
        venue = values[6]
        meta = metadata_processing(party,state,venue,job,subject,speaker)
        stmt = stmt
        word_id, pos_id, dep_id = stmt_processing(stmt)
        filepath = "model/lstm"
        label_pred, percent_pred = predict(filepath, word_id, pos_id, dep_id, meta)
        sg.popup("prediction: ",label_pred, "correct probability： ",percent_pred)
    if (event == "Feedback"):
        sg.popup("Please send your feedback to:", "yangwq177@gmail.com")

window.close()