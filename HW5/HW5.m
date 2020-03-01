% Cat Le
% HW 5
%% Problem 1
close all
clc, clear
%%
D = 2;
P = 50;
Q = 50;

yP = round(rand(P,D));
yQ = round(rand(Q,D));
Y = [yP;yQ];
sne = tsne(Y);

figure(1)
plot(sne(1:P,1),sne(1:P,2),'r*','LineWidth',3)
hold on, plot(sne(P+1:P+Q,1),sne(P+1:P+Q,2),'b*','LineWidth',3)
legend('Pennies','Quarters')

HH = [];
HT = [];
TH = [];
TT = [];
for i = 1:(P+Q)
    if sum(Y(i,:)==[1,1]) == 2
        HH = [HH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0]) == 2
        HT = [HT;sne(i,:)];
    elseif sum(Y(i,:)==[0,1]) == 2
        TH = [TH;sne(i,:)];
    else
        TT = [TT;sne(i,:)];
    end
end

figure(2)
plot(HH(:,1),HH(:,2),'b*','LineWidth',3)
hold on, plot(HT(:,1),HT(:,2),'c*','LineWidth',3)
hold on, plot(TH(:,1),TH(:,2),'k*','LineWidth',3)
hold on, plot(TT(:,1),TT(:,2),'r*','LineWidth',3)
legend('HH','HT','TH','TT')

%% Problem 2
close all
clc, clear
D = 3;
P = 50;
Q = 50;

yP = round(rand(P,D));
yQ = round(rand(Q,D));
Y = [yP;yQ];
sne = tsne(Y);

figure(1)
plot(sne(1:P,1),sne(1:P,2),'r*','LineWidth',3)
hold on, plot(sne(P+1:P+Q,1),sne(P+1:P+Q,2),'b*','LineWidth',3)
legend('Pennies','Quarters')

HHH = [];
HHT = [];
HTH = [];
HTT = [];
THH = [];
THT = [];
TTH = [];
TTT = [];
for i = 1:(P+Q)
    if sum(Y(i,:)==[1,1,1]) == 3
        HHH = [HHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0]) == 3
        HHT = [HHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1]) == 3
        HTH = [HTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0]) == 3
        HTT = [HTT;sne(i,:)];
        
    elseif sum(Y(i,:)==[0,1,1]) == 3
        THH = [THH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0]) == 3
        THT = [THT;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1]) == 3
        TTH = [TTH;sne(i,:)];
    else
        TTT = [TTT;sne(i,:)];
    end
end

figure(2)
plot(HHH(:,1),HHH(:,2),'b*','LineWidth',3)
hold on, plot(HHT(:,1),HHT(:,2),'c*','LineWidth',3)
hold on, plot(HTH(:,1),HTH(:,2),'k*','LineWidth',3)
hold on, plot(HTT(:,1),HTT(:,2),'r*','LineWidth',3)

hold on, plot(THH(:,1),THH(:,2),'bs','LineWidth',1)
hold on, plot(THT(:,1),THT(:,2),'cs','LineWidth',1)
hold on, plot(TTH(:,1),TTH(:,2),'ks','LineWidth',1)
hold on, plot(TTT(:,1),TTT(:,2),'rs','LineWidth',1)

legend('HHH','HHT','HTH','HTT','THH','THT','TTH','TTT')


%% Problem 3
close all
clc, clear
D = 4;
P = 50;
Q = 50;

yP = round(rand(P,D));
yQ = round(rand(Q,D));
Y = [yP;yQ];
sne = tsne(Y);

figure(1)
plot(sne(1:P,1),sne(1:P,2),'r*','LineWidth',3)
hold on, plot(sne(P+1:P+Q,1),sne(P+1:P+Q,2),'b*','LineWidth',3)
legend('Pennies','Quarters')

HHHH = [];
HHHT = [];
HHTH = [];
HHTT = [];
HTHH = [];
HTHT = [];
HTTH = [];
HTTT = [];
THHH = [];
THHT = [];
THTH = [];
THTT = [];
TTHH = [];
TTHT = [];
TTTH = [];
TTTT = [];

for i = 1:(P+Q)
    if sum(Y(i,:)==[1,1,1,1]) == 4
        HHHH = [HHHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,1,0]) == 4
        HHHT = [HHHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0,1]) == 4
        HHTH = [HHTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0,0]) == 4
        HHTT = [HHTT;sne(i,:)];   
    elseif sum(Y(i,:)==[1,0,1,1]) == 4
        HTHH = [HTHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1,0]) == 4
        HTHT = [HTHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0,1]) == 4
        HTTH = [HTTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0,0]) == 4
        HTTT = [HTTT;sne(i,:)];
        
    elseif sum(Y(i,:)==[0,1,1,1]) == 4
        THHH = [THHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,1,0]) == 4
        THHT = [THHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0,1]) == 4
        THTH = [THTH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0,0]) == 4
        THTT = [THTT;sne(i,:)];   
    elseif sum(Y(i,:)==[0,0,1,1]) == 4
        TTHH = [TTHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1,0]) == 4
        TTHT = [TTHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,0,1]) == 4
        TTTH = [TTTH;sne(i,:)];
    else
        TTTT = [TTTT;sne(i,:)];
    end
end

figure(2)
plot(HHHH(:,1),HHHH(:,2),'b*','LineWidth',3)
hold on, plot(HHHT(:,1),HHHT(:,2),'c*','LineWidth',3)
hold on, plot(HHTH(:,1),HHTH(:,2),'k*','LineWidth',3)
hold on, plot(HHTT(:,1),HHTT(:,2),'r*','LineWidth',3)
hold on, plot(HTHH(:,1),HTHH(:,2),'bs','LineWidth',1)
hold on, plot(HTHT(:,1),HTHT(:,2),'cs','LineWidth',1)
hold on, plot(HTTH(:,1),HTTH(:,2),'ks','LineWidth',1)
hold on, plot(HTTT(:,1),HTTT(:,2),'rs','LineWidth',1)

hold on, plot(THHH(:,1),THHH(:,2),'b+','LineWidth',3)
hold on, plot(THHT(:,1),THHT(:,2),'c+','LineWidth',3)
hold on, plot(THTH(:,1),THTH(:,2),'k+','LineWidth',3)
hold on, plot(THTT(:,1),THTT(:,2),'r+','LineWidth',3)
hold on, plot(TTHH(:,1),TTHH(:,2),'b^','LineWidth',1)
hold on, plot(TTHT(:,1),TTHT(:,2),'c^','LineWidth',1)
hold on, plot(TTTH(:,1),TTTH(:,2),'k^','LineWidth',1)
hold on, plot(TTTT(:,1),TTTT(:,2),'r^','LineWidth',1)

legend('HHHH','HHHT','HHTH','HHTT','HTHH','HTHT','HTTH','HTTT',...
    'THHH','THHT','THTH','THTT','TTHH','TTHT','TTTH','TTTT')


%% Problem 5
close all
clc, clear
D = 5;
P = 500;
Q = 500;

yP = round(rand(P,D));
yQ = round(rand(Q,D));
Y = [yP;yQ];
sne = tsne(Y);

figure(1)
plot(sne(1:P,1),sne(1:P,2),'r*','LineWidth',3)
hold on, plot(sne(P+1:P+Q,1),sne(P+1:P+Q,2),'b*','LineWidth',3)
legend('Pennies','Quarters')

HHHHH = [];
HHHHT = [];
HHHTH = [];
HHHTT = [];
HHTHH = [];
HHTHT = [];
HHTTH = [];
HHTTT = [];
HTHHH = [];
HTHHT = [];
HTHTH = [];
HTHTT = [];
HTTHH = [];
HTTHT = [];
HTTTH = [];
HTTTT = [];

THHHH = [];
THHHT = [];
THHTH = [];
THHTT = [];
THTHH = [];
THTHT = [];
THTTH = [];
THTTT = [];
TTHHH = [];
TTHHT = [];
TTHTH = [];
TTHTT = [];
TTTHH = [];
TTTHT = [];
TTTTH = [];
TTTTT = [];

for i = 1:(P+Q)
    if sum(Y(i,:)==[1,1,1,1,1]) == 5
        HHHHH = [HHHHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,1,1,0]) == 5
        HHHHT = [HHHHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,1,0,1]) == 5
        HHHTH = [HHHTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,1,0,0]) == 5
        HHHTT = [HHHTT;sne(i,:)];   
    elseif sum(Y(i,:)==[1,1,0,1,1]) == 5
        HHTHH = [HHTHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0,1,0]) == 5
        HHTHT = [HHTHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0,0,1]) == 5
        HHTTH = [HHTTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,1,0,0,0]) == 5
        HHTTT = [HHTTT;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1,1,1]) == 5
        HTHHH = [HTHHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1,1,0]) == 5
        HTHHT = [HTHHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1,0,1]) == 5
        HTHTH = [HTHTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,1,0,0]) == 5
        HTHTT = [HTHTT;sne(i,:)];   
    elseif sum(Y(i,:)==[1,0,0,1,1]) == 5
        HTTHH = [HTTHH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0,1,0]) == 5
        HTTHT = [HTTHT;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0,0,1]) == 5
        HTTTH = [HTTTH;sne(i,:)];
    elseif sum(Y(i,:)==[1,0,0,0,0]) == 5
        HTTTT = [HTTTT;sne(i,:)];
        
        
    elseif sum(Y(i,:)==[0,1,1,1,1]) == 5
        THHHH = [THHHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,1,1,0]) == 5
        THHHT = [THHHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,1,0,1]) == 5
        THHTH = [THHTH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,1,0,0]) == 5
        THHTT = [THHTT;sne(i,:)];   
    elseif sum(Y(i,:)==[0,1,0,1,1]) == 5
        THTHH = [THTHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0,1,0]) == 5
        THTHT = [THTHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0,0,1]) == 5
        THTTH = [THTTH;sne(i,:)];
    elseif sum(Y(i,:)==[0,1,0,0,0]) == 5
        THTTT = [THTTT;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1,1,1]) == 5
        TTHHH = [TTHHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1,1,0]) == 5
        TTHHT = [TTHHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1,0,1]) == 5
        TTHTH = [TTHTH;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,1,0,0]) == 5
        TTHTT = [TTHTT;sne(i,:)];   
    elseif sum(Y(i,:)==[0,0,0,1,1]) == 5
        TTTHH = [TTTHH;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,0,1,0]) == 5
        TTTHT = [TTTHT;sne(i,:)];
    elseif sum(Y(i,:)==[0,0,0,0,1]) == 5
        TTTTH = [TTTTH;sne(i,:)];
    else
        TTTTT = [TTTTT;sne(i,:)];
    end
end

figure(2)
plot(HHHHH(:,1),HHHHH(:,2),'b*','LineWidth',3)
hold on, plot(HHHHT(:,1),HHHHT(:,2),'c*','LineWidth',3)
hold on, plot(HHHTH(:,1),HHHTH(:,2),'k*','LineWidth',3)
hold on, plot(HHHTT(:,1),HHHTT(:,2),'r*','LineWidth',3)
hold on, plot(HHTHH(:,1),HHTHH(:,2),'bs','LineWidth',1)
hold on, plot(HHTHT(:,1),HHTHT(:,2),'cs','LineWidth',1)
hold on, plot(HHTTH(:,1),HHTTH(:,2),'ks','LineWidth',1)
hold on, plot(HHTTT(:,1),HHTTT(:,2),'rs','LineWidth',1)

hold on, plot(HTHHH(:,1),HTHHH(:,2),'b+','LineWidth',3)
hold on, plot(HTHHT(:,1),HTHHT(:,2),'c+','LineWidth',3)
hold on, plot(HTHTH(:,1),HTHTH(:,2),'k+','LineWidth',3)
hold on, plot(HTHTT(:,1),HTHTT(:,2),'r+','LineWidth',3)
hold on, plot(HTTHH(:,1),HTTHH(:,2),'b^','LineWidth',1)
hold on, plot(HTTHT(:,1),HTTHT(:,2),'c^','LineWidth',1)
hold on, plot(HTTTH(:,1),HTTTH(:,2),'k^','LineWidth',1)
hold on, plot(HTTTT(:,1),HTTTT(:,2),'r^','LineWidth',1)


hold on, plot(THHHH(:,1),THHHH(:,2),'b*','LineWidth',3)
hold on, plot(THHHT(:,1),THHHT(:,2),'c*','LineWidth',3)
hold on, plot(THHTH(:,1),THHTH(:,2),'k*','LineWidth',3)
hold on, plot(THHTT(:,1),THHTT(:,2),'r*','LineWidth',3)
hold on, plot(THTHH(:,1),THTHH(:,2),'bs','LineWidth',1)
hold on, plot(THTHT(:,1),THTHT(:,2),'cs','LineWidth',1)
hold on, plot(THTTH(:,1),THTTH(:,2),'ks','LineWidth',1)
hold on, plot(THTTT(:,1),THTTT(:,2),'rs','LineWidth',1)

hold on, plot(TTHHH(:,1),TTHHH(:,2),'bx','LineWidth',2)
hold on, plot(TTHHT(:,1),TTHHT(:,2),'cx','LineWidth',2)
hold on, plot(TTHTH(:,1),TTHTH(:,2),'kx','LineWidth',2)
hold on, plot(TTHTT(:,1),TTHTT(:,2),'rx','LineWidth',2)
hold on, plot(TTTHH(:,1),TTTHH(:,2),'bv','LineWidth',1)
hold on, plot(TTTHT(:,1),TTTHT(:,2),'cv','LineWidth',1)
hold on, plot(TTTTH(:,1),TTTTH(:,2),'kv','LineWidth',1)
hold on, plot(TTTTT(:,1),TTTTT(:,2),'rv','LineWidth',1)

legend('HHHHH','HHHHT','HHHTH','HHHTT','HHTHH','HHTHT','HHTTH','HHTTT',...
    'HTHHH','HTHHT','HTHTH','HTHTT','HTTHH','HTTHT','HTTTH','HTTTT',...
    'THHHH','THHHT','THHTH','THHTT','THTHH','THTHT','THTTH','THTTT',...
    'TTHHH','TTHHT','TTHTH','TTHTT','TTTHH','TTTHT','TTTTH','TTTTT')


