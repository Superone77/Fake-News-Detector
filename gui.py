import PySimpleGUI as sg
from para_list import *
from pre_processing import *
from predict import predict

sg.change_look_and_feel("TanBlue")
font = ("Arial", 14)

layout = [
    [sg.Image("picture/pic.PNG", size=(500,100))],
    [sg.Multiline('Enter your text', size=(63,3))],
    [sg.Text("Subject", size=[7, 1]), sg.InputCombo(subjects_list, size=(18, 1)), sg.Text("Speaker", size=[7, 1]), sg.InputCombo(speakers_list, size=(18, 1))],
     [sg.Text("Job", size=[7, 1]), sg.InputCombo(jobs_list, size=(18, 1)),sg.Text("State", size=[7, 1]), sg.InputCombo(states_list, size=(18, 1))], \
         [sg.Text("Party", size=[7, 1]), sg.InputCombo(parties_list, size=(18, 1)),sg.Text("Venue", size=[7, 1]), sg.InputCombo(venues_list, size=(18, 1))],
    [sg.Button("Check", size = (30,1)),sg.Button("Exit", size = (10,1)), sg.Button("Feedback", size = (10,1))]]
window = sg.Window('Fake News Detector', layout, size=(500, 320), resizable=True,font = font)
window.finalize()
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    if(event == "Check"):
        print("Checking")
        stmt = values[1]
        subject = values[2]
        speaker = values[3]
        job = values[4]
        state = values[5]
        party = values[6]
        venue = values[7]
        meta = metadata_processing(party,state,venue,job,subject,speaker)
        stmt = stmt
        word_id, pos_id, dep_id = stmt_processing(stmt)
        filepath = "model/lstm"
        label_pred, percent_pred = predict(filepath, word_id, pos_id, dep_id, meta)
        sg.popup("prediction: ",label_pred, "correct probability： ",percent_pred, title = "Result", font = font)
    if(event == "Feedback"):
        sg.popup("Please send your feedback to:", "yangwq177@gmail.com",title = "Feedback", font = font)

window.close()