%
%DATA FLIP BECAUSE OF DATA IMPORT DOWNLOAD ORDER%
stocks = flip(data);
%

stockcol=2;
currentprice = stocks{end,stockcol};
numrows = size(stocks,1);
startamount=10000;

sma1 = zeros(numrows,1);
sma2 = zeros(numrows,1);
smadiff = zeros(numrows,1);

%tax rates
over1yeartax = 0.78;
under1yeartax = 0.65;

besttaxedreturn=-1;

astart =5; %lower sma1 limit
aend = 200; %upper sma1 limit
bstart = 5; %lower sma2 limit
bend = 200; %upper sma2 limit
inc = 5;

combinations = (((aend-astart)/inc)+1)*(((bend-bstart)/inc)+1);
iterations = 0;

for a = astart:inc:aend
    tic
    for i = a:numrows
        sma1(i,1) = mean(stocks{i-a+1:i,stockcol});
    end
    
    for b = bstart:inc:bend
        for i = b:numrows
            sma2(i,1) = mean(stocks{i-b+1:i,stockcol});
        end
        
        %buy and sell limits/positions
        buysells=zeros(numrows,1);
        pos = 0; %if i own already or not (1=own, 0=no position)
        %buy=1, sell=-1, neither=0
        for i = b:numrows-1
            smadiff(i,1) = sma1(i,1) - sma2(i,1);
            if smadiff(i,1) - smadiff(i-1,1) > 0 && pos == 0
                buysells(i,1) = 1; %buy
                pos = 1;
            elseif smadiff(i,1) - smadiff(i-1,1) < 0 && pos == 1
                buysells(i,1) = -1; %sell
                pos = 0;
            else
                buysells(i,1) = 0;
            end
        end
        
        %calculate and list trades
        trades = zeros(numrows,10);
        tradecount = 0;
        for i = 2:numrows
            lastbuy = 0;
            if buysells(i,1) == 1
                tradecount = tradecount+1;
                trades(tradecount,1) = tradecount;
                trades(tradecount,2) = 1;
                trades(tradecount,3) = datenum(stocks{i,1});
                trades(tradecount,4) = stocks{i,stockcol};
            elseif buysells(i,1) == -1
                tradecount = tradecount+1;
                trades(tradecount,1) = tradecount;
                trades(tradecount,2) = -1;
                trades(tradecount,3) = datenum(stocks{i,1});
                trades(tradecount,4) = stocks{i,stockcol};
            end
        end
        
        %calculate returns & put into col 5 (cum in 6), then hold times in col 7
        trades(1,9) = startamount; %starting amount
        for i = 2:2:tradecount
            trades(i,5) = (trades(i,4)-trades(i-1,4))/trades(i-1,4);
            trades(i,7) = trades(i,3)-trades(i-1,3);
        end
        
        %cumulative returns and profit/loss
        trades(2,6) = trades(2,5); %before tax, first one
        trades(2,9) = (trades(2,5)*trades(1,9))+trades(1,9); %before tax liquidity, first one
        trades(2,10) = startamount*trades(2,5); %before tax p/l, first one
        for i = 4:2:tradecount
            trades(i,6) = (trades(i-2,6)+1)*(trades(i,5)+1)-1;
            trades(i,9) = (trades(i,5)*trades(i-2,9))+trades(i-2,9);
            trades(i,10) = trades(i-2,9)*trades(i,5) + trades(i-2,10);
        end
        
        %calculate p/l differences between over1year and under1year
        %initialize
        under1yearpl = 0;
        over1yearpl = 0;
        if tradecount > 0
            %first one
            if trades(i,7) ~= 0
                if trades(i,7) < 365
                    under1yearpl = trades(2,9)-trades(1,9);
                else
                    over1yearpl = trades(2,9)-trades(1,9);
                end
            end
            %rest of them
            for i = 4:2:tradecount
                if trades(i,7) ~= 0
                    if trades(i,7) < 365
                        under1yearpl = (trades(i,9)-trades(i-2,9))+under1yearpl;
                    else
                        over1yearpl = (trades(i,9)-trades(i-2,9))+over1yearpl;
                    end
                end
            end
        end
        
        %only include last sale for cumulative taxed return
        %initialize
        taxcumpl = 0;
        endtaxedliquidity = 0;
        %just for calculation steps
        over1yearplcalc = 0;
        under1yearplcalc = 0;
        if tradecount > 1
            %only tax if return is positive
            if under1yearpl + over1yearpl > 0
                %tax proper amounts
                if under1yearpl > 0
                    under1yearplcalc = under1yearpl*under1yeartax;
                else
                    under1yearplcalc = under1yearpl;
                end
                if over1yearpl > 0
                    over1yearplcalc = over1yearpl*over1yeartax;
                else
                    over1yearplcalc = over1yearpl;
                end
                taxcumpl = under1yearplcalc + over1yearplcalc;
            end
        else
            %dont tax if return is negative
            taxcumpl = under1yearpl+over1yearpl;
        end
        taxcumreturn = 0;
        endtaxedliquidity = startamount+taxcumpl;
        taxcumreturn = (endtaxedliquidity/startamount)-1;
        
        %save best stats
        if taxcumreturn > besttaxedreturn
            besttaxedreturn = taxcumreturn;
            besta = a;
            bestb = b;
            besttradecount = tradecount;
            besttrades = trades;
            bestunder1yearpl = under1yearpl;
            bestover1yearpl = over1yearpl;
            bestendtaxedliquidity = endtaxedliquidity;
            %calculate number of losing trades
            losingtrades=0;
            losingtradepct = 0;
            for i = 1:besttradecount
                if besttrades(i,5)<0
                    losingtrades=losingtrades+1;
                end
            end
            losingtradepct=(losingtrades*2)/besttradecount;
        end
        iterations = iterations + 1;
    end
    
    %updating stats during run
    iterationtime=toc;
    outputstats1 = [besttaxedreturn, besta, bestb, besttradecount, iterations, combinations, a, b, taxcumreturn]
    
end


%return if not using algorthm
if datenum(stocks{end,1})-datenum(stocks{1,1}) < 365 && ((stocks{end,stockcol}-stocks{1,stockcol})/(stocks{1,stockcol})) > 0
    noalgoreturn = ((stocks{end,stockcol}-stocks{1,stockcol})/(stocks{1,stockcol}))*under1yeartax;
elseif datenum(stocks{end,1})-datenum(stocks{1,1}) > 364 && ((stocks{end,stockcol}-stocks{1,stockcol})/(stocks{1,stockcol})) > 0
    noalgoreturn = ((stocks{end,stockcol}-stocks{1,stockcol})/(stocks{1,stockcol}))*over1yeartax;
else
    noalgoreturn = ((stocks{end,stockcol}-stocks{1,stockcol})/(stocks{1,stockcol}));
end

%better off
if noalgoreturn<0
    betteroff = abs((besttaxedreturn/noalgoreturn)-1);
else
    betteroff = (besttaxedreturn/noalgoreturn)-1;
end

maxdrawdown0 = min(besttrades);
maxdrawdown = maxdrawdown0(1,5);

%turn besttrade array into table
besttrades = array2table(besttrades, 'VariableNames',{'TradeNumber','Buy/Sell','DateNum','Price','PreTaxReturn','PreTaxCumReturn','HoldTime','Date','PreTaxLiquidity','PreTax Running P/L'});
%turn datenum into datestr in col 8
%besttrades{:,8} = datestr(besttrades{:,3});
avgtradepct = besttaxedreturn/(besttradecount/2);

%output results
outputresults1 = [betteroff, besttaxedreturn, noalgoreturn, besta, bestb, besttradecount, avgtradepct, iterations, combinations]
outputresults2 = [startamount, bestendtaxedliquidity,(noalgoreturn+1)*startamount, losingtrades, losingtradepct, maxdrawdown]