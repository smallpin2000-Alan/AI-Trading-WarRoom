#D使用
import sys
# import msvcrt 


from PY_Trade_package.MarketDataMart import MarketDataMart, SystemEvent
from PY_Trade_package.Product import ProductBasic, ProductSnapshot, ProductTick
from PY_Trade_package.Sol_D import Sol_D
from PY_Trade_package.SolPYAPI_Model import RCode

import time
# import keyboard #hide PW

_detail = ""

def Observer_OnSystemEvent(data:SystemEvent):
    print(f"\n SystemEvent: [SystemEvent] {data}")

def Event_OnUpdateBasic(data):
    "ProductBasic data"          
    global _detail
    if _detail.upper() == "D":      
        print(f"\n UpdateBasic: {data}") #所有欄位資料
    else:
        print(f"""\n UpdateBasic: 交易所代碼:{data.Exchange} 商品代碼:{data.Symbol} 參考價:{data.TodayRefPrice} 漲停價:{data.RiseStopPrice} 跌停價:{data.FallStopPrice} 商品中文名稱:{data.ChineseName}
    上一交易日成交總量:{data.PreTotalMatchQty} 上一交易日參考價:{data.PreTodayRefPrice} 上一交易日收盤價:{data.PreClosePrice} 上一交易日成交總額:{data.PreTotalTradingAmount}""")
        

def Event_OnInterrupt(date):
    print(date)

def Event_OnMath(data):
    "TX event(ProductTick data)"
    global _detail
    if _detail.upper() == "D":      
        print(f"\n OnMath Symbol: {data}") #所有欄位資料
    else:
        print(f"""\n OnMath Symbol:{data.Symbol} MatchTime:{data.MatchTime} 
    MatchPrice:{data.MatchPrice} MatchQty:{data.MatchQty} 
    DayHigh:{data.DayHigh} DayLow:{data.DayLow} OpenPrice:{data.OpenPrice}
    BuyPrice:{data.BuyPrice} SellPrice:{data.SellPrice}""")    

def Event_OnOrderBook(data):
    "5Q(ProductTick data)"    
    global _detail
    if _detail.upper() == "D":      
        print(f"\n OnOrder Symbol: {data}") #所有欄位資料
    else:
        print(f"""\n OnOrder Symbol:{data.Symbol} OrderBookTime:{data.OrderBookTime}
    DayHigh:{data.DayHigh} DayLow:{data.DayLow} OpenPrice:{data.OpenPrice}
    BuyPrice:{data.BuyPrice} SellPrice:{data.SellPrice} """)    

def QuoteEvent_OnUpdateBasicList(data:list):
    "List<ProductSnapshot> data"
    for snapshot in data:
        if _detail.upper() == "D":      
            print(f"\n LastSnapshotList TickData: {snapshot.TickData}") #所有欄位資料
        else:  
            print(f"""\n LastSnapshotList TickData: Symbol:{snapshot.TickData.Symbol} MatchTime:{snapshot.TickData.MatchTime}
        MatchPrice:{snapshot.TickData.MatchPrice} MatchQty:{snapshot.TickData.MatchQty} 
        DayHigh:{snapshot.TickData.DayHigh} DayLow:{snapshot.TickData.DayLow} OpenPrice:{snapshot.TickData.OpenPrice} RefPrice:{snapshot.TickData.RefPrice}
        BuyPrice:{snapshot.TickData.BuyPrice} SellPrice:{snapshot.TickData.SellPrice}""")
  
def Observer_OnUpdateLastSnapshot(snapshot:ProductSnapshot):
    try:
        global _detail
        if _detail.upper() == "D":      
            print(f"\n LastSnapshot TickData: {snapshot.TickData}") #所有欄位資料
        else:  
            print(f"""\n LastSnapshot TickData: Symbol:{snapshot.TickData.Symbol} MatchTime:{snapshot.TickData.MatchTime}
        MatchPrice:{snapshot.TickData.MatchPrice} MatchQty:{snapshot.TickData.MatchQty} 
        DayHigh:{snapshot.TickData.DayHigh} DayLow:{snapshot.TickData.DayLow} OpenPrice:{snapshot.TickData.OpenPrice} RefPrice:{snapshot.TickData.RefPrice}
        BuyPrice:{snapshot.TickData.BuyPrice} SellPrice:{snapshot.TickData.SellPrice}""")        
       
    except Exception as ex:
        pass
 #region 驗證EVENT
def QuoteApi_OnSystemEvent_DAPI(data:SystemEvent):
    "驗證失敗訊息"
    try:
        msg:str = f"[{data.rcode}]{data.msg}"
        print(msg)
    except Exception as ex:
        pass
def QuoteApi_OnAnnouncementEvent_DAPI(data:str):
    "公告"
    try:        
        print(f"公告: {data}")
    except Exception as ex:
        pass
def QuoteApi_OnLoginResultEvent_DAPI(aIsSucc:bool, aMsg:str):
    "登入是否成功"
    try:        
        print(f"IsSucc:{aIsSucc}, Msg:{aMsg}")
    except Exception as ex:
        pass
def QuoteApi_OnVerifiedEvent_DAPI(data):
    "驗證成功與否"
    try:        
        print(f"MarketKind:{data.MarketKind}, IsSucc:{data.IsSucc}, Msg:{data.Msg}")
    except Exception as ex:
        pass    
def QuoteApi_OnUpdateBasic_DAPI(data:ProductBasic):
    "傳來驗證商品資料"    
    print(f"""\n UpdateBasic: 交易所代碼:{data.Exchange} 商品代碼:{data.Symbol} 參考價:{data.TodayRefPrice} 漲停價:{data.RiseStopPrice} 跌停價:{data.FallStopPrice} 商品中文名稱:{data.ChineseName}""")
    # print(f"[驗證成交商品資料]:{data}") #所有欄位資料

def QuoteApi_OnMatch_DAPI(data:ProductTick):
    "傳來驗證成交行情"
    try:        
        print(f"""[驗證成交行情] 商品代碼:{data.Symbol} 累計成交量:{data.TotalMatchQty} 成交價:{data.MatchPrice} 單筆成交量:{data.MatchQty}""")
        #print(f"[驗證成交行情]{data}")  #所有欄位資料
    except Exception as ex:
        pass
 #endregion 驗證EVENT

# def masked_input(prompt='請輸入密碼：'):
#         password = ''
#         print(prompt, end='', flush=True)

#         while True:
#             char = msvcrt.getch()
#             if char == b'\r' or char == b'\n':
#                 print()
#                 break
#             elif char == b'\x03':  # Ctrl+C
#                 raise KeyboardInterrupt
#             elif char == b'\x08':  # Backspace
#                 if len(password) > 0:
#                     password = password[:-1]
#                     print('\b \b', end='', flush=True)
#             else:
#                 char = char.decode('utf-8')
#                 password += char
#                 print('*', end='', flush=True)

#         return password  

class Sample_D:
    _id:str=""
    _pwd:str=""
    _type:str=""
    _prod:str=""
    _detail:str = ""
    _sec:int = 0
    def run():
        fQuoteEvent = MarketDataMart()
        fQuoteEvent.OnSystemEvent = Observer_OnSystemEvent #系統訊息通知        
        fQuoteEvent.OnUpdateBasic = Event_OnUpdateBasic #商品基本資料
        fQuoteEvent.OnMatch = Event_OnMath  #商品成交行情
        fQuoteEvent.OnOrderBook = Event_OnOrderBook #商品五檔行情        
        fQuoteEvent.OnUpdateLastSnapshot = Observer_OnUpdateLastSnapshot #國內國內商品最新快照更新 
       
        solD = Sol_D(fQuoteEvent,__file__)        
        solD.Set_OnSystemEvent_DAPI(QuoteApi_OnSystemEvent_DAPI) #驗證失敗訊息
        solD.Set_OnUpdateBasic_DAPI(QuoteApi_OnUpdateBasic_DAPI) #傳來驗證商品資料
        solD.Set_OnMatch_DAPI(QuoteApi_OnMatch_DAPI) #傳來驗證成交行情
        solD.Set_OnLogEvent(Sample_D.funLog)#回傳錯誤通知
        solD.Set_OnInterruptEvent(Event_OnInterrupt)#server回傳中斷事件
        solD.Set_OnAnnouncementEvent_DAPI(QuoteApi_OnAnnouncementEvent_DAPI)#公告
        solD.Set_OnLoginResultEvent_DAPI(QuoteApi_OnLoginResultEvent_DAPI)#登入是否成功
        solD.Set_OnVerifiedEvent_DAPI(QuoteApi_OnVerifiedEvent_DAPI)#驗證成功與否
        IsWaitVerify=True
        while (IsWaitVerify):
            #登入
            rc = solD.Login(Sample_D._id, Sample_D._pwd, Sample_D._type) #登入帳號,密碼
            if rc == RCode.OK:#判斷回傳OK
                IsWaitVerify = False
                print("\n connect success.")        
                
                conn:bool = solD.IsConnect()
                print(f"check connect:{conn}")
                
                start = time.time()
                rc = solD.Subscribe(Sample_D._type, Sample_D._prod)#訂閱商品
                print(f"\n Subscribe return->{rc}")
                                    
                ct = 1        
                while (True):
                    end = time.time()
                    duration = end - start
                    if float(Sample_D._sec) < duration:
                        break
                    print(f"{int(duration)}sec", end='', flush=True)
                    time.sleep(1)                   
                    ct+=1
                    
                if rc == RCode.OK:
                    rc = solD.Unsubscribe(Sample_D._type, Sample_D._prod)
                    print(f"\n Unsubscribe return: {Sample_D._prod} ->{rc}")   
            else:
                answer = input("輸入 'Y' 重新驗證 或'N'結束 \n")
                if answer.upper() == 'Y':
                    continue
                else:
                    break
    
        rc = solD.DisConnect()
        print(f"\n DisConnect->{rc}")        
        sys.exit(0)

    def funLog(msg:str):
        print(msg)

if __name__ == '__main__':    
    print(f"get args: {sys.argv}")
    if len(sys.argv) == 0 or len(sys.argv) < 6:
        print("please key in ID, password, TWS, product code, sec.")
        sys.exit(0)
    Sample_D._id = sys.argv[1]
    Sample_D._pwd = sys.argv[2]
    Sample_D._type = sys.argv[3]
    Sample_D._prod = sys.argv[4]
    Sample_D._sec = sys.argv[5]
    if len(sys.argv) >= 7:
        _detail = sys.argv[6]
        
    Sample_D.run()
    
            




                     