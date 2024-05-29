# Federated-Face-Recognition


Go to terminal and enter "pip install ." within this folder directory.

enter "pip install tensorflow"


FOR CLIENTS


edit path of dataset in line 13
edit path of local model in line 34
update server address in line 91

Modify the "dataset.py" on how you load your dataset.

run "python client.py --client_id=1" number depends on client.

FOR SERVER
edit path of global model and global model rounds
run "python server.py"



TESTING 
"pip install cv"
"pip install keras-facenet"

To try testing the global and local model

Go to test_1 folder and edit test.py path of model and dataset
run "python test.py"

Client_1.py is variation of test.py but it applies gaussian blur and can save the image.



