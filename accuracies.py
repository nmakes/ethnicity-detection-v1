import matplotlib.pyplot as plt
import numpy as np

def plotGraphs(trloss, tracc, teloss, teacc, trainingTimes, testingTimes):

    size = len(trloss)

    plt.figure('Loss over epochs')
    line_trainloss, = plt.plot(range(1,size+1), trloss, label='Train Loss')
    line_testloss, = plt.plot(range(1,size+1), teloss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(handles=[line_trainloss, line_testloss])

    plt.figure('Accuracies over epochs')
    line_trainacc, = plt.plot(range(1,size+1), tracc, label='Train Accuracy')
    line_testacc, = plt.plot(range(1,size+1), teacc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(handles=[line_trainacc, line_testacc])

    plt.figure('Time per image over epochs')
    line_traintime, = plt.plot(range(1,size+1), np.array(trainingTimes) * 1000, label='Training time per image')
    line_testtime, = plt.plot(range(1,size+1), np.array(testingTimes) * 1000, label='Testing time per image')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    plt.legend(handles=[line_traintime, line_testtime])

    plt.show()

# VERSION 1
trloss = [1.5914696075782577, 1.1481950188793728, 0.9957214363360017, 0.9245880912901469, 0.8588065761493971, 0.8210342868573455, 0.7800862056837102, 0.7566242481059835, 0.7335255051769802, 0.6923306552939279, 0.674795988199355, 0.6530991100000889, 0.6338260278283837, 0.6197981561870616, 0.6068594013844536, 0.5851974186267438, 0.5577788787899374, 0.5440768756787543, 0.5229572814701161, 0.5010301084961221, 0.4884794580216719, 0.4748229224222261, 0.45821582743115413, 0.4398848171198328, 0.43365921750961545, 0.4023378663383815, 0.40095026886539886, 0.38811199179354744, 0.36771018167890873, 0.3549960228131295]
tracc = [0.411048114939113, 0.5728623967137104, 0.6344621518693597, 0.662044131277238, 0.6920012259571018, 0.7041832663842662, 0.7201195219488843, 0.7257891513984919, 0.7362090101133926, 0.7544437639741783, 0.7577382781883343, 0.7669322712268707, 0.777275513349557, 0.7765093472082119, 0.7863162733132657, 0.7905301874742674, 0.8006435789617702, 0.8078455405479336, 0.8145111856780606, 0.8216365307085433, 0.8231688631921686, 0.8300643584642752, 0.8322862401316949, 0.8426294825831842, 0.8462304632027311, 0.8559607721858327, 0.8569567886700924, 0.8580294203765546, 0.866457247566015, 0.8734293591015331]
teloss = [1.102677031757087, 1.0657104261147763, 1.1193227283810323, 0.8564869519195005, 0.8184102116539521, 0.7831694765792753, 0.750423264388697, 0.8029592983742364, 0.730426540278011, 0.6759212151108287, 0.6536052676309746, 0.6494715795474334, 0.6444749732948891, 0.6721955110643064, 0.6616588605278296, 0.6151123751144462, 0.6387742004745436, 0.642223450676775, 0.630362617038959, 0.6456311047692083, 0.6907206779087098, 0.6533607724526234, 0.6517222780718928, 0.6423157767137766, 0.6634435775437414, 0.673595479678009, 0.7029933090305066, 0.7049200729539995, 0.667243144904075, 0.7031485309627066]
teacc = [0.607290233755702, 0.6024759283092032, 0.5598349381837753, 0.6836313618426474, 0.7083906462464733, 0.7111416779653242, 0.7269601102052398, 0.7180192572214581, 0.7359009626970986, 0.7627235210745338, 0.7771664371680689, 0.7806052270420972, 0.7799174691328812, 0.7661623107026007, 0.7771664374140302, 0.7867950478970922, 0.7771664374960174, 0.7833562584329996, 0.7902338377711206, 0.7861072903158248, 0.7654745527933847, 0.7806052267141487, 0.7819807425325805, 0.7854195323246217, 0.7806052270420972, 0.7819807427785419, 0.7744154058591536, 0.7792297112236652, 0.7874828058063083, 0.7764786796687886]
trainingTimes = [0.001271883001219375, 0.0010729608015527471, 0.0010760056618127289, 0.0010728506160739592, 0.0010724724380289745, 0.001070699003478555, 0.0010716987101899455, 0.001075569778872603, 0.0010747211935912544, 0.0010745174089093314, 0.0010762630230877028, 0.001073796143387126, 0.0010736035928015982, 0.0010734240664611592, 0.0010758212582304403, 0.0010719048878257346, 0.0010745546732290807, 0.0010718607186467378, 0.0010738623058215045, 0.001071643946253373, 0.001070465242939422, 0.0010705964717987744, 0.0010736538996332597, 0.0010759289959548918, 0.0010498053377989132, 0.0010334453150205294, 0.00103101611977664, 0.001031956879448898, 0.0010327677072062655, 0.0010314459563982184]
testingTimes = [0.0005876157109970239, 0.000510539280499192, 0.0005022774372336953, 0.000502288587483612, 0.0005081359097193491, 0.0005034257490008537, 0.0005003517234833729, 0.0005052865289428375, 0.000503621862219977, 0.0005095552709440418, 0.0004983166388993086, 0.0005030635298528238, 0.0005043799152400503, 0.0005029366137728894, 0.0005009830243977752, 0.0005004173131887654, 0.0004976943565693977, 0.0004997322287159412, 0.0005021460938486469, 0.0005044559992983056, 0.0005068278870195602, 0.0005017689530426403, 0.0005019334192289119, 0.0005048947944273811, 0.0005198359653415182, 0.0004970036969716152, 0.0004914869468510561, 0.0005036769575725067, 0.0005086981774688259, 0.0004972931115466594]

plotGraphs(trloss, tracc, teloss, teacc, trainingTimes, testingTimes)
print('mean TeAcc', np.max(teacc), np.argmax(teacc) + 1)
print('mean Tr Time', np.mean(trainingTimes) * 1000)
print('mean Te Time', np.mean(testingTimes) * 1000)
print('\n')

# VERSION 2
trloss = [1.4739660322392165, 0.9056463600154891, 0.7727541327841838, 0.7089398310405872, 0.6528205529444429, 0.6156393858942926, 0.5701201900433369, 0.5391007008415185, 0.5070832625437908, 0.46818745434010206, 0.42820080657185056, 0.40194489342684836, 0.37313584076857736, 0.3446621316691465, 0.31935620144434296, 0.28858341294455814, 0.26653506842230934, 0.24803340947412472, 0.23388237671694287, 0.2106106127793534, 0.19212733419201616, 0.18238603398363684, 0.17554166183272937, 0.15716878109426904, 0.1431791898076643, 0.14131072302409706, 0.14130855012151897, 0.12107543857814416, 0.12107077150294797, 0.11378727422657825]
tracc = [0.4944069870330058, 0.6756818875227463, 0.7306159978405377, 0.751072633076474, 0.7706864850856466, 0.7798038611848991, 0.7994177137420764, 0.8087649407687817, 0.8179589340082531, 0.8339718050142756, 0.8469966293806099, 0.8546582904104583, 0.8654612324152334, 0.8748084586016649, 0.8881397485148493, 0.8971038916671419, 0.9063745015536281, 0.9105884157146298, 0.914802328688288, 0.9271376029574867, 0.9311982841532617, 0.9338798655383689, 0.9384002446616825, 0.9452191235425097, 0.9503524367370161, 0.948973337784889, 0.950122586357568, 0.9581673307320913, 0.9577842479536879, 0.9603125958255636]
teloss = [1.093570845081193, 0.9334067046396342, 0.7683148038518478, 0.889714439231723, 0.9423161765773161, 0.6991285903909675, 0.7533098759316676, 0.7049279236727929, 0.7501655962968955, 0.7581732779782281, 0.7258530315703506, 0.8250512015868742, 0.8205780786202075, 0.7823051266689904, 0.8888621571303398, 0.822658951787542, 0.9059276344687608, 1.0226965445763798, 0.8674785361165358, 0.9627240878187806, 1.0069905234170256, 0.9452230223599294, 0.996524131445642, 1.0870951550206915, 1.0951555067112226, 1.0664653423891763, 1.0083031487924032, 1.1696276696708867, 1.1584309864240958, 1.22783603202526]
teacc = [0.6086657495741339, 0.6533700137551581, 0.7414030262167877, 0.6911966988440229, 0.680192571968619, 0.7544704264918909, 0.7400275101523944, 0.7558459423923098, 0.7572214579647804, 0.7537826683367135, 0.7682255844302486, 0.7482806053909343, 0.7530949107554461, 0.7496561211273791, 0.7469050892445539, 0.7530949104274975, 0.7654745530393462, 0.7503438787906337, 0.7757909214316241, 0.7592847316104412, 0.75240715284623, 0.752407152764243, 0.7682255844302486, 0.7599724895196571, 0.7606602474288731, 0.7579092157920092, 0.7530949104274975, 0.7579092157920092, 0.7579092157920092, 0.7537826685826748]
trainingTimes = [0.0011272397539639743, 0.0010196922419446032, 0.001018057763119837, 0.0010223700303882315, 0.0010225965024648251, 0.0010197001332123148, 0.001019876426325011, 0.0010186107546648218, 0.001020696716296903, 0.001018177027209858, 0.0010177180002044761, 0.0010162428994139317, 0.0010178069596344659, 0.0010160533811210892, 0.00101525873776926, 0.0010163530666258962, 0.00101429834126184, 0.001015669375959438, 0.0010149742685283505, 0.001014217218299092, 0.001014369161736187, 0.001013919249875685, 0.0010178454478313834, 0.0010156634209750074, 0.0010139495180020697, 0.0010128285013164364, 0.001012139860340835, 0.0010129948938108462, 0.0010147451660291863, 0.0010158491397685811]
testingTimes = [0.0005873126865581108, 0.0005204810400940529, 0.0005041775709989147, 0.0005103608765005246, 0.0005122026354279447, 0.0005016825386057858, 0.0005042480799322116, 0.0004878309766724808, 0.0005104684436173681, 0.0005262212871357368, 0.000488798260852756, 0.00048644359042916696, 0.0004780769676093058, 0.0005251992355514589, 0.0005040458996653392, 0.0005433485629306043, 0.0005124394142644114, 0.0004735655436981495, 0.0004798083718574031, 0.0004928112030029297, 0.00047193416375077577, 0.0004904175067046322, 0.0004872590344414586, 0.0005053970475964238, 0.0004843379969118058, 0.0004785039565914107, 0.0005256609870774218, 0.0005229585272409893, 0.0005041001751465515, 0.0005073096434056677]

plotGraphs(trloss, tracc, teloss, teacc, trainingTimes, testingTimes)
print('mean TeAcc', np.max(teacc), np.argmax(teacc) + 1)
print('mean Tr Time', np.mean(trainingTimes) * 1000)
print('mean Te Time', np.mean(testingTimes) * 1000)
print('\n')

# VERSION 3
trloss = [1.4956181108166096, 0.9792932090750509, 0.8598023672575896, 0.7969206651587666, 0.7419525017922471, 0.7027972660801216, 0.6564258132797788, 0.6223171317354507, 0.5950325683439724, 0.5624508551392422, 0.5392085061075496, 0.5060077908212263, 0.48035424853664505, 0.4520476355239945, 0.42328790743548994, 0.3965636730157558, 0.36721273739755134, 0.33823261816663097, 0.31462661639298906, 0.2972450111765749, 0.28103761599175664, 0.2616686232792173, 0.24836833366323313, 0.23173557392684294, 0.2171410432201275, 0.205464109828638, 0.2057522827501937, 0.17918913319382682, 0.16437346736437844, 0.15557374545138122]
tracc = [0.47356726956667067, 0.63844621526731, 0.6913116758782331, 0.7124578611747714, 0.7329144953146983, 0.751532332520159, 0.7675452035079147, 0.7785779959286717, 0.7863162733315325, 0.7982684651693975, 0.8079987743351673, 0.8193380327046447, 0.8255439778248153, 0.8417100832756772, 0.8473797118484773, 0.8587189701448874, 0.8660741650616139, 0.8788691383908944, 0.8882163651618642, 0.8954183268210949, 0.9007814898653115, 0.906680968269555, 0.9110481153409831, 0.9144192456176155, 0.9230769231682572, 0.9288997856013584, 0.9252221882982631, 0.9355654301104135, 0.9429206252646088, 0.9465216053909514]
teloss = [0.9546382652680189, 0.8601940968259657, 0.8344260542314023, 0.7324602869140725, 0.6989768643490714, 0.7116171884897486, 0.6529062806822082, 0.6967362040420674, 0.6644320388771645, 0.6768920618369458, 0.6409102977961917, 0.6451402388953769, 0.6223725437134136, 0.7466883271973596, 0.6940634868413906, 0.7057075843604428, 0.6917635608654075, 0.7159232532879674, 0.732481216138134, 0.7569683643449944, 0.7306486821158225, 0.7885347698626852, 0.8053891136934865, 0.7777423657774105, 0.8366797216328827, 0.8485833719024304, 0.8314688435692899, 0.8307225004842882, 0.9287022793104429, 0.9587467075050123]
teacc = [0.6629986243202073, 0.697386519699018, 0.6994497936726273, 0.7407152683075717, 0.7469050894905154, 0.7530949104274975, 0.7799174691328812, 0.7448418157628675, 0.7778541954052334, 0.7696011004946419, 0.7785419533144493, 0.7861072899878763, 0.7840440162602285, 0.7448418157628675, 0.7799174688869198, 0.7689133422574774, 0.7723521321315057, 0.7744154058591536, 0.7737276476219891, 0.7709766163130738, 0.7895460795339561, 0.7867950478970922, 0.7702888584038579, 0.7757909216775855, 0.7799174691328812, 0.7792297108957167, 0.7792297112236652, 0.7867950482250408, 0.7647867948021817, 0.782668500769745]
trainingTimes = [0.0018781183725334258, 0.0016842837937215045, 0.0016869131750831055, 0.0016852541273808573, 0.0016866330533461668, 0.0016848073756808048, 0.0016841579170414104, 0.0016860589818869709, 0.0016840787120951197, 0.0016843458643874397, 0.001683919863798777, 0.0016934935972802375, 0.0016946212082891204, 0.0016932085617678025, 0.0016924567175832146, 0.0016932280159347305, 0.0016903671208532756, 0.001691019264715711, 0.0016907627071809667, 0.0017507030413700985, 0.001754402822049673, 0.0017590925184232927, 0.0017592882474360932, 0.0017484612820011832, 0.0017610650430820195, 0.0017596599224919453, 0.0017481307803652896, 0.0017463716122039505, 0.0017427756236149715, 0.0017420038320592236]
testingTimes = [0.0007770983028280685, 0.0006662075916379485, 0.0006682941641407459, 0.0007127682477276788, 0.0006693331050741623, 0.000644879131238431, 0.0007279291441548805, 0.0006611727618777604, 0.0006636481173592716, 0.0007000741801202871, 0.0006714013124594498, 0.0006633509959938438, 0.0007098808249220395, 0.0006938361072146745, 0.000690356573343605, 0.0006630630571871709, 0.0006786971833387792, 0.0006682472675013902, 0.0006721936361005907, 0.0006460836861779634, 0.0006410668935867582, 0.0006491083554421364, 0.0006533382355428956, 0.000655798013469376, 0.0006933425446815963, 0.0006414307524774228, 0.0006448594543268133, 0.0006626462346094019, 0.0006484104809767607, 0.0006535763261734702]

plotGraphs(trloss, tracc, teloss, teacc, trainingTimes, testingTimes)
print('mean TeAcc', np.max(teacc), np.argmax(teacc) + 1)
print('mean Tr Time', np.mean(trainingTimes) * 1000)
print('mean Te Time', np.mean(testingTimes) * 1000)
print('\n')

# VERSION 4
trloss = [1.4100213822067569, 1.091484237988595, 0.9399409222887807, 0.8689211813211807, 0.8227644013803209, 0.7946789679701183, 0.7656431589578143, 0.7403194099673821, 0.721649326632953, 0.7129832158744829, 0.6998497857807825, 0.6772604915688749, 0.6628677178262167, 0.6596728985820342, 0.6520167267874198, 0.6360719512840959, 0.6322117894659589, 0.6174616833570505, 0.6250037470186849, 0.6044202947631191, 0.6041110138603781, 0.5885408626098119, 0.5890778467365104, 0.57346752285044, 0.5766754756301158, 0.5627987371826698, 0.5649034248478311, 0.554671102338432, 0.5513613123836014, 0.5413212749697188]
tracc = [0.4302022679156041, 0.5903309836476777, 0.6598988664029677, 0.6850291143850501, 0.7068648477693734, 0.7139135759336394, 0.7260956170184094, 0.737511492820375, 0.7447900704872816, 0.7490806001627551, 0.7530646641452438, 0.7593472267709698, 0.7650934722465201, 0.7673153536582045, 0.7661661045009871, 0.7775053631079332, 0.7781949120907925, 0.7828685258416139, 0.779037695646359, 0.788691388493917, 0.7862396567393182, 0.788538155035486, 0.7875421396655025, 0.7961998155538633, 0.7956635003582379, 0.797655531390474, 0.7987281638824095, 0.8020992953281186, 0.8050873428263475, 0.8043977931128153]
teloss = [1.2669225970521425, 1.1078126715006822, 0.98211751072738, 0.9255561698253741, 0.895187178454996, 0.7939870858044867, 0.8020230468562414, 0.8410900281908752, 0.7080162069164575, 0.6632598910731823, 0.6662592551321243, 0.7269707521349397, 0.6292110361784, 0.6446019540626704, 0.6159515768658508, 0.6577852455261157, 0.6263565780618003, 0.628537826352795, 0.5833742926832406, 0.6319496473387151, 0.7386961794539035, 0.5821481255668394, 0.6158193398620601, 0.6419950468258155, 0.5684168386123203, 0.5769383726946425, 0.5903974718207998, 0.628267081682423, 0.592188677359808, 0.6240469193524146]
teacc = [0.5392022007433224, 0.6079779915829309, 0.6595598346921404, 0.6506189822002815, 0.681568087787051, 0.7145804674294168, 0.7166437414850133, 0.6932599724076965, 0.7434662996984741, 0.7668500686118167, 0.7709766163130738, 0.753782668664662, 0.7806052271240843, 0.7640990369749528, 0.7771664375780045, 0.7806052271240843, 0.777166437250056, 0.7833562584329996, 0.7971114166173188, 0.7854195324066089, 0.7420907838800422, 0.7943603848984677, 0.7888583216247401, 0.7792297112236652, 0.8081155430827869, 0.7991746905909279, 0.8005502064093599, 0.8012379643185759, 0.8081155434107354, 0.7867950479790794]
trainingTimes = [0.002189397483086827, 0.001770266864038924, 0.0017737837024820826, 0.0018050145817942829, 0.0018066309216634064, 0.0018060550417886934, 0.0018054710514443877, 0.0018054262429265716, 0.0017792903277320873, 0.0017849654826948404, 0.0017747702570806206, 0.0017821081496439497, 0.0017742333038065867, 0.0017845285768126038, 0.0017864756557863255, 0.0017888105028873202, 0.0017818799604859558, 0.0017759877956611337, 0.0017737514067383, 0.001791379439330269, 0.0017780627058650539, 0.0017862307159512679, 0.0017727280810242447, 0.0017563126732372924, 0.0017585455914636788, 0.0017592722274319657, 0.0017561461528751187, 0.0017570261936263735, 0.0017631348203085626, 0.0017930649191259863]
testingTimes = [0.0009192038107115105, 0.0006734414802456821, 0.0006692581688357514, 0.0006955555413415376, 0.0006911726732529506, 0.0006810569369645361, 0.0006735965998989352, 0.0007163346879583934, 0.000672897905562242, 0.000676969878447269, 0.0006622917222517557, 0.0006619462284786009, 0.0006701403503732799, 0.0007103542186207096, 0.0006760978633140763, 0.0006657218998695174, 0.0006791128580967039, 0.0006830664415634974, 0.0006679039053936608, 0.0006657504313913632, 0.0006734009786026022, 0.0006972018429468881, 0.0006708565899561655, 0.0006692409515380859, 0.0006732906239232794, 0.0006684253435515308, 0.0006683389291146762, 0.0006664648672573504, 0.000679394401907101, 0.0007321391833697585]

plotGraphs(trloss, tracc, teloss, teacc, trainingTimes, testingTimes)
print('mean TeAcc', np.max(teacc), np.argmax(teacc) + 1)
print('mean Tr Time', np.mean(trainingTimes) * 1000)
print('mean Te Time', np.mean(testingTimes) * 1000)
print('\n')