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

class Station:
    def __init__(self, name, port=None, type=None):
        self.name = name
        self.port = port
        self.type = type


class Kray:
    def __init__(self, name, ports):
        self.name = name
        self.ports = ports


class Port:
    """
        :param
        name - назание порта
        kray - край, в котором находится порт
        coordinates - координаты порта
        """
    def __init__(self, name, kray, coordinates = None):
        self.name = name
        self.kray = kray
        self.coordinates = coordinates


class Task:
    """
    :param
    i - id заявки
    from_st - станция отправления
    to_st - станция назначения
    j - нить
    k - id корабля
    """
    def __init__(self, appl_id, from_st, to_st, ship_id):
        self.i = appl_id
        self.from_st = from_st
        self.to_st = to_st
        self.j = f'{self.from_st.name} - {self.to_st.name}'
        self.k = ship_id


class Destination:
    """
    params:
    thread_name - имя нити
    distance - расстояние между пунктами
    time - время в пути
    freight_charge - стоимость провоза 1 контейнера
    """
    def __init__(self, graph, from_st, to_st, thread_name, distance, time, freight_charge):
        self.graph = graph
        self.thread_name = thread_name
        self.from_st = from_st
        self.to_st = to_st
        self.distance = distance
        self.time = time
        self.freight_charge = freight_charge


class Ship:
    """
    params:
    ship_id - id корабля
    port - название порта
    load_start - время начала погрузки
    load_end - время конца погразки
    container_capacity - вместимость корабля (число контейнеров)
    freight_charge_filled - оплата контейнероместа при использовании
    freight_charge_vacant - оплата контейнероместа (резервирование)

    """
    def __init__(self, ship_id, port, load_start, load_end,
                 container_capacity, freight_charge_filled, freight_charge_vacant):
        self.ship_id = ship_id
        self.port = port
        self.load_start = load_start
        self.load_end = load_end
        self.container_capacity = container_capacity
        self.freight_charge_filled = freight_charge_filled
        self.freight_charge_vacant = freight_charge_vacant


class Application:

    """
    params:
    appl_id - номер заявки
    start - начало
    end - конец
    num_cont - число контейнеров
    departure - точка отправления
    arrival - точка прибытия
    cont_cost - стоимость провоза (прибыль)
    """

    def __init__(self, appl_id, start, end, num_cont, departure, arrival, cont_cost):
        self.appl_id = appl_id
        self.start = start
        self.end = end
        self.num_cont = num_cont
        self.departure = departure
        self.arrival = arrival
        self.cont_cost = cont_cost


if __name__ == "__main__":

    krays = np.array([Kray('Приморский', ['Находка', 'Восточный', 'Владивосток']),
                    Kray('Хабаровский', ['Ванино'])])
    ports = np.array([Port(name='Находка', kray='Приморский', coordinates="42°47' С.Ш. 132°52' В.Д."),
                      Port(name='Восточный', kray='Приморский', coordinates="42°46' С.Ш. 133°03' В.Д."),
                      Port(name='Владивосток', kray='Приморский', coordinates="43°05' С.Ш. 131°54' В.Д."),
                      Port(name='Ванино', kray='Хабаровский', coordinates="53°00' С.Ш. 158°39' В.Д."),
                      Port(name='Зарубино', kray='Приморский', coordinates="42°38′20″ С.Ш. 131°04′30″ В.Д.")])
    stations = [Station('Хани'), Station('Беркакит'), Station('Бестужево'), Station('Тында'),
                         Station('Штурм'), Station('Новый Ургал'), Station('Известковая'),
                         Station('Архара'), Station('Комсомольск-на-Амуре'), Station('Волочаевка'),
                         Station('Биробиджан'), Station('Ленинск'), Station('Хабаровск'),
                         Station('Сибирцево'), Station('Новочугуевка'), Station('Новокачалинск'),
                         Station('Гродеково'), Station('Уссурийск'), Station('Барановский'),
                         Station('Хасан'),
                         Station(name='Ванино', port=ports[3]),
                         Station(name='Посьет', port=ports[4]),
                         Station(name='Владивосток', port=ports[2]),
                         Station(name='Дунай', port=ports[0]),
                         Station(name='Находка', port=ports[1])]
    threads = [(stations[0], stations[3], {'weight': 486}), (stations[3], stations[4], {'weight': 162}),
               (stations[2], stations[3], {'weight': 27}), (stations[1], stations[2], {'weight': 193}),
               (stations[5], stations[2], {'weight': 924}), (stations[5], stations[6], {'weight': 347}),
               (stations[8], stations[5], {'weight':  538}), (stations[6], stations[7], {'weight': 154}),
               (stations[8], stations[9], {'weight': 353}), (stations[6], stations[10], {'weight': 116}),
               (stations[10], stations[11], {'weight': 121}), (stations[10], stations[9], {'weight': 124}),
               (stations[8], stations[20], {'weight': 489}), (stations[9], stations[12], {'weight': 49}),
               (stations[13], stations[12], {'weight': 586}), (stations[13], stations[14], {'weight': 163}),
               (stations[13], stations[15], {'weight': 127}), (stations[13], stations[17], {'weight': 68}),
               (stations[17], stations[16], {'weight': 97}), (stations[17], stations[18], {'weight': 23}),
               (stations[18], stations[24], {'weight': 223}), (stations[18], stations[23], {'weight': 155}),
               (stations[18], stations[22], {'weight': 89}),
               (stations[18], stations[19], {'weight': 237}), (stations[18], stations[21], {'weight': 195})]

    #граф
    G = nx.Graph()
    G.add_nodes_from(stations)
    G.add_edges_from(threads)
    # print(G[stations[18]][stations[19]]['weight'])

    #считает в км длину наименьшего пути
    # print(nx.shortest_path_length(G, source=stations[0], target=stations[19], weight='weight', method='dijkstra'))

    #выводит наименьший путь
    # print([_.name for _ in nx.shortest_path(G, source=stations[0], target=stations[19], weight='string', method='dijkstra')])

    # def func(el):
    #     return el[0].name, el[1].name
    # print(list(map(func, G.edges())))
    # train_schedule = pd.DataFrame(columns=['train_id', 'from', 'to', 'stations', 'time_departure', 'time_arrival'])
    # port_schedule = pd.DataFrame(columns=['ship_id', 'start_load', 'end_load', 'capacity_available', 'time_arrival'])

    port_schedule = pd.DataFrame(columns=['port', 'time_dep', 'cost'])
    train_data = pd.read_csv('charge_trains.csv', sep=';')

    random.seed(10)

    for i in range(100):

        port_ = random.choice(['Ванино', 'Посьет', 'Владивосток', 'Дунай', 'Находка'])
        start_date = datetime.datetime.today().date() #datetime.date(year, month, day)
        end_date = datetime.date(2022, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)

        time_ = random_date
        cost_ = random.randint(30000, 50000)
        list_ = [port_, time_, cost_]
        # lists.append(pd.Series(list_))
        port_schedule.loc[i] = list_

    # Step 0: Create an instance of the model
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)
    # Step 1: Define index sets
    to_port = list(port_schedule['port'].unique())
    time_dep = list(port_schedule['time_dep'].unique())
    from_st = 'Тында'
    cost_downtime = 10000 #стоимость простоя
    curr_date = datetime.datetime.today()
    # Step 2: Define the decision
    model.x = Var(to_port, time_dep, within=Binary)
    # Step 3: Define Objective
    model.Cost = Objective(
        expr=sum([(train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['стоимость, руб'].to_numpy()[0]
                 + ((j - curr_date.date()).days - train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0]) * cost_downtime + port_schedule[(port_schedule['port'] == i) & (port_schedule['time_dep'] == j)]['cost'].to_numpy()[0]) * model.x[i, j] for i in to_port for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique())]),
        sense=minimize)
    # Step 4: Constraints
    model.x_weight = Constraint(
        expr=sum(model.x[i, j] for i in to_port for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique())) >= 1)

    model.time_weight = Constraint(
        expr=sum(((j - curr_date.date()).days -
                  train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0])
                  * model.x[i, j] for i in to_port for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique())) >= 1)
    # model.dmd = ConstraintList()
    # for i in to_port:
    #     for j in list(port_schedule[port_schedule['port'] == i]['time_dep'].unique()):
    #         train_days = train_data[(train_data['от'] == from_st) & (train_data['до'] == i)]['время, дней'].to_numpy()[0]
    #         # print((j - curr_date.date()).days - train_days)
    #         model.dmd.add(((j - curr_date.date()).days - train_days) * model.x[i, j] >= 0)

    # solve
    results = SolverFactory('glpk').solve(model)
    # results.write()
    # print(train_data[(train_data['от'] == from_st) & (train_data['до'] == 'Ванино')]['стоимость, руб'].to_numpy()[0] + (list(port_schedule[port_schedule['port'] == 'Ванино']['time_dep'].unique())[0] - curr_date.date()).days)

    for k in to_port:
        for j in list(port_schedule[port_schedule['port'] == k]['time_dep'].unique()):
            t = train_data[(train_data['от'] == from_st) & (train_data['до'] == k)]['стоимость, руб'].to_numpy()[0]
            p = port_schedule[(port_schedule['port'] == k) & (port_schedule['time_dep'] == j)]['cost'].to_numpy()[0]
            if model.x[k, j]() == float(1):
                train_days = train_data[(train_data['от'] == from_st) & (train_data['до'] == k)]['время, дней'].to_numpy()[0]
                print(f'со станции {from_st}, в пути {train_days} дней')
                print(f'из порта {k}, отправляется {j}: {model.x[k, j]()}')
                print(f'sum: {t + p}')
                print(f"train: {t}")
                print(f"port: {p}")
                print(f"стоимость простоя: {cost_downtime * ((j - curr_date.date()).days - train_days)}")
                # print(port_schedule[port_schedule['port'] == k])









