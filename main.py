# оптимизация
from pyomo.environ import *
# работа с данными
import numpy as np
import pandas as pd
import csv
# для работы с графами
import networkx as nx
# визуализация
import seaborn as sns
from matplotlib import pyplot as plt
#решатель
import glpk
#дополнения
import random
import datetime

if __name__ == "__main__":

    port_schedule = pd.DataFrame(columns=['port', 'time_dep', 'cost'])
    task_schedule = pd.DataFrame(columns=['id_task', 'from_station', 'time_st', 'time_end', 'num_cont', 'payment'])
    train_data = pd.read_csv('charge_trains.csv', sep=';')

    random.seed(10)

    for i in range(100):
        #наполняем port_schedule

        port_ = random.choice(['Ванино', 'Посьет', 'Владивосток', 'Дунай', 'Находка'])
        start_date = datetime.datetime.today().date() #datetime.date(year, month, day)
        end_date = datetime.date(2022, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)

        time_ = random_date
        cost_ = random.randint(30000, 50000)

        # list_ = [порт отправления, время отправления, стоимость провоза]
        list_ = [port_, time_, cost_]
        port_schedule.loc[i] = list_

        # наполняем task_schedule

        # станция, с которой необходимо отправить груз
        task_from = random.choice(train_data['от'].unique())
        # груз может быть отправлен от
        task_st_date = datetime.datetime.today().date()  # datetime.date(year, month, day)
        # груз должен быть на корабле до
        task_end_date = task_st_date + datetime.timedelta(days=random.randrange(1, 100))
        # число контейнеров
        task_num_cont = random.randrange(1, 10)
        # оплата провоза, предлагаемая в заявке
        task_payment = random.randrange(50000, 150000)
        list_task = [i, task_from, task_st_date, task_end_date, task_num_cont, task_payment]
        task_schedule.loc[i] = list_task

    # from_st = input(f"Выберите станцию отправления из списка: {train_data['от'].unique()}: ")
    result_tasks = pd.DataFrame(columns=['id заявки', 'От станции', 'До станции(порт)', 'Время отправления корабля', 'Доход от заявки', 'Расходы на выполнение', 'Прибыль'])
    cnt = 0
    for st in train_data['от'].unique():
        from_st = st
        print(from_st)
        # Step 0: Create an instance of the model
        model = ConcreteModel()
        model.dual = Suffix(direction=Suffix.IMPORT)
        # Step 1: Define index sets
        to_port = list(port_schedule['port'].unique())
        time_dep = list(port_schedule['time_dep'].unique())
        ID_task = list(task_schedule[task_schedule['from_station'] == from_st]['id_task'])
        cost_downtime = 10000 #стоимость простоя
        curr_date = datetime.datetime.today()
        # Step 2: Define the decision
        model.x = Var(to_port, time_dep, ID_task, within=Binary)
        # Step 3: Define Objective
        model.Cost = Objective(
            expr=sum([(task_schedule[task_schedule['id_task'] == k]['payment'].to_numpy()[0]
                       - (train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['стоимость, руб'].to_numpy()[0]
                     + ((j - curr_date.date()).days - train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0])
                          * cost_downtime + port_schedule[(port_schedule['port'] == i) & (port_schedule['time_dep'] == j)]['cost'].to_numpy()[0])) * model.x[i, j, k]
                      for i in to_port for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique()) for k in ID_task]),
            sense=maximize)
        # Step 4: Constraints

        model.x_weight = Constraint(
            expr=sum(model.x[i, j, k] for i in to_port for j in
                     list(port_schedule[port_schedule['port'] == i]['time_dep'].unique()) for k in ID_task) <= 1)

        model.time_cost = Constraint(
            expr=sum((task_schedule[task_schedule['id_task'] == k]['payment'].to_numpy()[0] -
                      train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['стоимость, руб'].to_numpy()[0] -
                      port_schedule[(port_schedule['port'] == i) & (port_schedule['time_dep'] == j)]['cost'].to_numpy()[0] -
                      cost_downtime * ((j - curr_date.date()).days -
                                       train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0]))
                      * model.x[i, j, k] for i in to_port for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique())
                     for k in ID_task) >= 0)

        model.time_weight = Constraint(
            expr=sum((task_schedule[task_schedule['id_task'] == k]['time_end'].to_numpy()[0] - j).days
                     * model.x[i, j, k] for i in to_port for j in
                     list(port_schedule[port_schedule['port'] == i]['time_dep'].unique())
                     for k in ID_task) >= 0)

        model.time_id = Constraint(
            expr=sum(((j - curr_date.date()).days -
                      train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0])
                     * model.x[i, j, k] for i in to_port for j in
                     list(port_schedule[port_schedule['port'] == i]['time_dep'].unique()) for k in ID_task) >= 1)

        # solve
        results = SolverFactory('glpk').solve(model)
        # results.write()

        for i in to_port:
            for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique()):
                for k in ID_task:
                    t = train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['стоимость, руб'].to_numpy()[0]
                    p = port_schedule[(port_schedule['port'] == i) & (port_schedule['time_dep'] == j)]['cost'].to_numpy()[0]
                    if model.x[i, j, k]() == float(1):
                        flag = 1
                        train_days = train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0]
                        # print(f'со станции {from_st}, в пути ~ {train_days} дней')
                        # print(f'Прибывает на станцию {i}, корабль отправляется {j}: {model.x[i, j, k]()}')
                        # print(f"выручка: {task_schedule[task_schedule['id_task'] == k]['payment'].to_numpy()[0]}")
                        # print(f'стоимость провоза на поезде и корабле: {t + p}')
                        # print(f"train: {t}")
                        # print(f"port: {p}")
                        income = task_schedule[task_schedule['id_task'] == k]['payment'].to_numpy()[0]
                        expense = t + p + cost_downtime * ((j - curr_date.date()).days - train_days)
                        profit = income - expense
                        # print(f"прибыль: {profit}")
                        # print(f"стоимость простоя: {cost_downtime * ((j - curr_date.date()).days - train_days)}")
                        # print(f"id заявки: {k}")
                        # print(task_schedule[task_schedule['id_task'] == k]['time_end'].to_numpy()[0])
                        # print((task_schedule[task_schedule['id_task'] == k]['time_end'].to_numpy()[0] - j).days
                      # * model.x[i, j, k])
                      #   columns = ['id заявки', 'От станции', 'До станции(порт)', 'Время отправления корабля',
                      #              'Доход от заявки', 'Расходы на выполнение', 'Прибыль']
                        result_tasks.loc[cnt] = [k, from_st, i, j, income, expense, profit]
                        cnt += 1
        # if flag == 0:
        #     print("Нет заявок, удовлетворяющих условиям")

        # print(task_schedule[task_schedule['from_station'] == 'Беркакит']['time_end'].to_numpy()[0])
    print(result_tasks)
    result_tasks.to_csv('results.csv')






