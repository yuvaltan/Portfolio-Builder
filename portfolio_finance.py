import datetime as dt
from datetime import timedelta
import yfinance as yf
import numpy as np

class PortfolioBuilder:

    def rec_permutation(self, data, perm_length):
        if perm_length == 1:
            return [[atom] for atom in data]
        res_list = []  # how can we make it better ?
        smaller_perm = self.rec_permutation(data, perm_length - 1)
        for elem in data:
            for sub_combination in smaller_perm:
                res_list.append([elem] + sub_combination)
        return res_list

    def all_portfolios(self, a, stock_num):  # creating a list of all possible portfolio vectors (omega)
        lst = [i for i in range(a + 1)]
        x = self.rec_permutation(lst, stock_num)
        y = []
        for i in range(len(x)):
            if (sum(x[i]) == a):
                y.append(np.asarray(x[i], dtype=np.float64) / a)  # every company gets k/a.. not k..
        return y

    def all_xt(self, closearray):   # getting array of adj close- and return list of all vector-xt (if 5 days- 4 vectors)
        all_xt = []
        for i in range(len(closearray) - 1):
            all_xt.append(closearray[i + 1] / closearray[i])
        return all_xt

#######################################################################################################
    def get_daily_data(self, tickers_list, start_date, end_date=dt.date.today()):
        try:
            data = yf.download(tickers_list, start_date + timedelta(days=1), end_date + timedelta(days=1), threads=False)
            closedata = data['Adj Close']
            self.list_of_closing = closedata.values
            self.list_of_all_xt = self.all_xt(self.list_of_closing)
            self.number_of_companies = len(self.list_of_closing[0])
            self.number_of_days = len(self.list_of_closing)
            return closedata
        except:
            raise ValueError
#######################################################################################################

    def wealth2(self, s0, portfolio, lst, num):  # recursive- creating a list of all profets for a single bw (also can return the last profet)
        if (num == 1):
            return s0
        else:
            st = self.wealth2(s0, portfolio, lst, num - 1) * (portfolio.dot(self.list_of_all_xt[num - 2]))
            lst.append(st)
            return st

    def profets_of_every_bw(self):  # returning a list of all the wealths of all portfolios in omega
        lstall = []
        for i in range(len(self.omega)):
            list1 = []
            self.wealth2(1, self.omega[i], list1, self.number_of_days)
            lstall.append(list1)
        return lstall

    def algorithem1(self, a):
        self.omega = self.all_portfolios(a, self.number_of_companies)
        list_of_profets_for_every_bw = self.profets_of_every_bw()
        b1 = np.asarray([1 / self.number_of_companies for i in range(self.number_of_companies)])  # creating a list with initial b1
        list_of_bt = [b1, b1]
        for j in range(len(list_of_profets_for_every_bw[0]) - 1):  # for every profet each day
            mone = 0
            mechane = 0
            for i in range(len(list_of_profets_for_every_bw)):  # for each possibble bw vector that day
                st_b = list_of_profets_for_every_bw[i][j]  # the wealth of this portfolio in that day
                mone = mone + st_b * self.omega[i]  # sums all the bw*st(bw)
                mechane += st_b  # sums all the st(bw)
            list_of_bt.append(mone / mechane)  # the result (bt vector) - inside the list
        return list_of_bt

    def final_wealth(self, list_of_bt, s0=1):  # wealth of the bt vectors
        list_of_profets = [1.0]
        st = s0
        for i in range(len(self.list_of_all_xt)):
            st = st * (list_of_bt[i + 1].dot(self.list_of_all_xt[i]))
            list_of_profets.append(st)
        return list_of_profets
    def final_wealth2(self, list_of_bt, s0=1):  # wealth of the bt vectors
        list_of_profets = [1.0]
        st = s0
        for i in range(len(self.list_of_all_xt)):
            st = st * (list_of_bt[i].dot(self.list_of_all_xt[i]))
            list_of_profets.append(st)
        return list_of_profets

#######################################################################
    def find_universal_portfolio(self, portfolio_quantization = 20):
        list_of_bt_al1 = self.algorithem1(portfolio_quantization)
        return self.final_wealth(list_of_bt_al1)
#######################################################################

    def calculate_bt(self, n, bt, xt):  # n is the input argument of the real method
        bt_xt = bt.dot(xt)
        vector_b = []
        mechane = 0
        for k in range(len(bt)):  # to calculate the mechane whitch depends on all the vector xt, bt
            exp_argument = (n * xt[k]) / bt_xt
            mechane += bt[k] * (np.exp(exp_argument))  # sigma((bt)k*e^(whatever))
        for j in range(len(bt)):
            exp_argument = (n * xt[j]) / bt_xt
            mone = bt[j] * (np.exp(exp_argument))
            vector_b.append(mone / mechane)
        return np.asarray(vector_b, dtype=np.float64)

#############################################################################################################
    def find_exponential_gradient_portfolio(self, learn_rate=0.5):
        list_of_bt2 = [np.asarray([1 / self.number_of_companies for i in range(self.number_of_companies)])]  # initial bt vector
        for i in range(len(self.list_of_all_xt)):
            list_of_bt2.append(self.calculate_bt(learn_rate, list_of_bt2[len(list_of_bt2) - 1], self.list_of_all_xt[i]))
        return self.final_wealth2(list_of_bt2)
############################################################################################################








