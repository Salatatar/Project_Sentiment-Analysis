from django.shortcuts import render
from django.http import HttpResponse
import pyrebase
import locale

firebaseConfig = {
    'apiKey': "AIzaSyCErkdR0G1y05dq5Ea2pavPbC-gTHeyssY",
    'authDomain': "webssru-87cc4.firebaseapp.com",
    'databaseURL': "https://webssru-87cc4.firebaseio.com",
    'projectId': "webssru-87cc4",
    'storageBucket': "webssru-87cc4.appspot.com",
    'messagingSenderId': "231310531528",
    'appId': "1:231310531528:web:4f48608234c255b70d3efd",
    'measurementId': "G-1SNRWYBJD0"
}

firebase = pyrebase.initialize_app(firebaseConfig)
locale.setlocale(locale.LC_ALL, 'en_US')
db = firebase.database()

def financial(request):    
    return render(request, 'financial.html', {})

def financial_info(request):
    if request.method == "POST":
        income = request.POST.get('income')
        fee = request.POST.get('fee')
        savings = request.POST.get('savings')
        debt = request.POST.get('debt')
        sports = request.POST.get('sports')
        price = request.POST.get('price')
        data = {"income": income,
                "fee": fee,
                "savings": savings,
                "debt": debt,
                "sports": sports, 
                "price": price}
        db.child("Requirement_financial").push(data)
    else:
        return render(request, 'financial.html', {})
    return render(request, 'financial.html', {'alert_flag': True})

def condition(request):
    if request.method == "POST":
        condition_G1 = request.POST.get('G1_optionsRadios')
        condition_G2 = request.POST.get('G2_optionsRadios')
        condition_G3 = request.POST.get('G3_optionsRadios')
        condition_G4 = request.POST.get('G4_optionsRadios')
        data = {"condition_G1": condition_G1,
                "condition_G2": condition_G2,
                "condition_G3": condition_G3,
                "condition_G4": condition_G4,}
        db.child("Condition_financial").push(data)
    else:
        return render(request, 'condition.html', {})
    return render(request, 'condition.html', {})

def result(request):
    requirement_financial = db.child("Requirement_financial").get()
    for require in requirement_financial.each():
        message_fi = require.val()
    income = float(message_fi['income'])
    fee = float(message_fi['fee'])
    savings = float(message_fi['savings'])
    debt = float(message_fi['debt'])
    sports = message_fi['sports']
    price = float(message_fi['price'])

    requirement_codition = db.child("Condition_financial").get()
    for require in requirement_codition.each():
        message_co = require.val()
    condition_G1 = message_co['condition_G1']
    condition_G2 = message_co['condition_G2']
    condition_G3 = message_co['condition_G3']
    condition_G4 = message_co['condition_G4']

    # t1 คือ เงินคงเหลือ = รายได้ - รายจ่าย
    # t2 คือ เงินออม = เงินคงเหลือ * 20 / 100
    # t3 คือ เงินเก็บ = เงินคงเหลือ - เงินออม
    # 30%ของเงินเดือน = รายได้ * 30 / 100
    t1 = income - fee
    t2 = ((t1 * 20) / 100)
    t3 = t1 - t2
    slrdebt = ((income * 30) / 100)
    count_G1 = 0
    if condition_G1 == "G1_option1":
        if t1 >= price:
            count_G1 = 75
        elif t3 >= price:
            count_G1 = 62.5
        elif (t2 * 30) >= price:
            count_G1 = 50
        elif slrdebt >= debt:
            count_G1 = 25
        elif slrdebt <= debt:
            count_G1 = 0
    elif condition_G1 == "G1_option2":
        if t1 >= price:
            count_G1 = 87.5
        elif t3 >= price:
            count_G1 = 75
        elif (t2 * 30) >= price:
            count_G1 = 62.5
        elif slrdebt >= debt:
            count_G1 = 37.5
        elif slrdebt <= debt:
            count_G1 = 12.5
    elif condition_G1 == "G1_option3":
        if t1 >= price:
            count_G1 = 100
        elif t3 >= price:
            count_G1 = 87.5
        elif (t2 * 30) >= price:
            count_G1 = 75
        elif slrdebt >= debt:
            count_G1 = 52.5
        elif slrdebt <= debt:
            count_G1 = 50

    count_G2 = 0
    if condition_G2 == "G2_option1":
        if t1 >= price:
            count_G2 = 75
        elif t3 >= price:
            count_G2 = 50
        elif (t2 * 30) >= price:
            count_G2 = 25
        elif slrdebt >= debt:
            count_G2 = 0
        elif slrdebt <= debt:
            count_G2 = -25
    elif condition_G2 == "G2_option2":
        if t1 >= price:
            count_G2 = 87.5
        elif t3 >= price:
            count_G2 = 62.5
        elif (t2 * 30) >= price:
            count_G2 = 37.5
        elif slrdebt >= debt:
            count_G2 = 12.5
        elif slrdebt <= debt:
            count_G2 = -12.5
    elif condition_G2 == "G2_option3":
        if t1 >= price:
            count_G2 = 100
        elif t3 >= price:
            count_G2 = 75
        elif (t2 * 30) >= price:
            count_G2 = 50
        elif slrdebt >= debt:
            count_G2 = 25
        elif slrdebt <= debt:
            count_G2 = 0  

    count_G3 = 0
    if condition_G3 == "G3_option1":
        if t1 >= price:
            count_G3 = 75
        elif t3 >= price:
            count_G3 = 62.5
        elif (t2 * 30) >= price:
            count_G3 = 50
        elif slrdebt >= debt:
            count_G3 = 25
        elif slrdebt <= debt:
            count_G3 = 0
    elif condition_G3 == "G3_option2":
        if t1 >= price:
            count_G3 = 87.5
        elif t3 >= price:
            count_G3 = 75
        elif (t2 * 30) >= price:
            count_G3 = 62.5
        elif slrdebt >= debt:
            count_G3 = 37.5
        elif slrdebt <= debt:
            count_G3 = 12.5
    elif condition_G3 == "G3_option3":
        if t1 >= price:
            count_G3 = 100
        elif t3 >= price:
            count_G3 = 87.5
        elif (t2 * 30) >= price:
            count_G3 = 75
        elif slrdebt >= debt:
            count_G3 = 62.5
        elif slrdebt <= debt:
            count_G3 = 50

    count_G4 = 0
    if condition_G4 == "G4_option1":
        if count_G1 >= 87.5 or count_G2 >= 100 or count_G3 >= 87.5:
            count_G4 = 100
        elif count_G1 >= 75 or count_G2 >= 87.5 or count_G3 >= 75:
            count_G4 = 87.5
        elif count_G1 >= 62.5 or count_G2 >= 75 or count_G3 >= 62.5:
            count_G4 = 75
        elif count_G1 >= 50 or count_G2 >= 62.5 or count_G3 >= 50:
            count_G4 = 50
        elif count_G1 >= 37.5 or count_G2 >= 50 or count_G3 >= 37.5:
            count_G4 = 50
        elif count_G1 >= 25 or count_G2 >= 37.5 or count_G3 >= 25:
            count_G4 = 37.5
        elif count_G1 >= 12.5 or count_G2 >= 25 or count_G3 >= 12.5:
            count_G4 = 25
        elif count_G1 >= 0 or count_G2 >= 12.5 or count_G3 >= 0:
            count_G4 = 12.5
        elif count_G1 >= 0 or count_G2 >= 0 or count_G3 >= 0:
            count_G4 = 0
        elif count_G1 >= 0 or count_G2 >= -12.5 or count_G3 >= 0:
            count_G4 = -12.5
        elif count_G1 >= 0 or count_G2 >= -25 or count_G3 >= 0:
            count_G4 = -25
    elif condition_G4 == "G4_option2":
        if count_G1 >= 62.5 or count_G2 >= 87.5 or count_G3 >= 62.5:
            count_G4 = 100
        elif count_G1 >= 50 or count_G2 >= 75 or count_G3 >= 50:
            count_G4 = 87.5
        elif count_G1 >= 37.5 or count_G2 >= 62.5 or count_G3 >= 37.5:
            count_G4 = 75
        elif count_G1 >= 25 or count_G2 >= 50 or count_G3 >= 25:
            count_G4 = 62.5
        elif count_G1 >= 12.5 or count_G2 >= 37.5 or count_G3 >= 12.5:
            count_G4 = 50
        elif count_G1 >= 0 or count_G2 >= 25 or count_G3 >= 0:
            count_G4 = 37.5
        elif count_G1 >= 0 or count_G2 >= 12.5 or count_G3 >= 0:
            count_G4 = 25
        elif count_G1 >= 0 or count_G2 >= 0 or count_G3 >= 0:
            count_G4 = 12.5
        elif count_G1 >= 0 or count_G2 >= -12.5 or count_G3 >= 0:
            count_G4 = 0
        elif count_G1 >= 0 or count_G2 >= -25 or count_G3 >= 0:
            count_G4 = -12.5
    elif condition_G4 == "G4_option3":
        if count_G1 >= 37.5 or count_G2 >= 62.5 or count_G3 >= 37.5:
            count_G4 = 100
        elif count_G1 >= 25 or count_G2 >= 50 or count_G3 >= 25:
            count_G4 = 87.5
        elif count_G1 >= 12.5 or count_G2 >= 37.5 or count_G3 >= 12.5:
            count_G4 = 75
        elif count_G1 >= 0 or count_G2 >= 25 or count_G3 >= 0:
            count_G4 = 62.5
        elif count_G1 >= 0 or count_G2 >= 12.5 or count_G3 >= 0:
            count_G4 = 50
        elif count_G1 >= 0 or count_G2 >= 0 or count_G3 >= 0:
            count_G4 = 37.5
        elif count_G1 >= 0 or count_G2 >= -12.5 or count_G3 >= 0:
            count_G4 = 25
        elif count_G1 >= 0 or count_G2 >= -25 or count_G3 >= 0:
            count_G4 = 12.5

    results = ((count_G1 + count_G2 + count_G3 + count_G4) / 4)
    _results = "%.2f"%(results)
    resultss = _results + " %"
    _price = str(locale.format("%.2f", price, grouping=True))
    prices = "{} บาท".format(_price)
    return render(request, 'result.html', {"results":results, "resultss":resultss, "products":sports.capitalize(), "prices":prices})
