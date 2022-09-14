echo "  Begin: l1" >> ../process.log;
cd ./exp-l1;
python3 -W ignore run.py --epoch_num=100 --batch_size=128 --reg_rate=0.05 --train --test
cd ..;
echo "  End: l1" >> ../process.log;

echo "  Begin: l2" >> ../process.log;
cd ./exp-l2;
python3 -W ignore run.py --epoch_num=100 --batch_size=128 --reg_rate=0.05 --train --test
cd ..;
echo "  End: l2" >> ../process.log;

echo "  Begin: rq" >> ../process.log;
cd ./exp-rq;
python3 -W ignore run.py --epoch_num=100 --batch_size=128 --reg_rate=0.05 --train --test
cd ..;
echo "  End: rq" >> ../process.log;

echo "  Begin: none" >> ../process.log;
cd ./exp-none;
python3 -W ignore run.py --epoch_num=100 --batch_size=128 --reg_rate=0.05 --train --test
cd ..;
echo "  End: none" >> ../process.log;
