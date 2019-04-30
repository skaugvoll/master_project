import os, sys



def cmd_input(question, funcYes, funcNo, yesPrint="Done", noPrint="Exiting"):
    asw = ''
    acceptable_yes_answers = ['yes', 'y']
    acceptable_no_answers = ['no', 'n']

    while asw not in acceptable_yes_answers + acceptable_no_answers:
        asw = input(question)

    if asw in acceptable_yes_answers:
        funcYes()
        print(yesPrint)
    elif asw in acceptable_no_answers:
        print(noPrint)
        funcNo()
