Code of FSRL in AAAI2020 paper: Few-shot Knowledge Graph Completion

Contact: Chuxu Zhang (chuxuzhang@gmail.com)


<1> How to use

Get data from: https://drive.google.com/uc?id=1ElKgnVcdoq7vdcC2N_N7lEeFP_Ughf_R&export=download

Get pretrained embedding from: https://github.com/xwhan/One-shot-Relational-Learning

Run the code: python main.py [parameter]

(Note that we set candidate size as 1000, you can find data processing code in data_process.py)

(default setting for NELL data, set dropout ratio ~0.1 for Wiki data)

(create models/ directory before running)

<2> If you find code useful, pls consider cite our paper.

@inproceedings{zhang2020kg,

  title={Few-Shot Knowledge Graph Completion},
  
  author={Chuxu Zhang and Huaxiu Yao and Chao Huang and Meng Jiang and Zhenhui Li and Nitesh V. Chawla},
  
  booktitle={Proceedings of AAAI Conference on Artificial Intelligence},
  
  year={2020}
  
}

