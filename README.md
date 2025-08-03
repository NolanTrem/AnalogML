# AnalogML
A project looking to implement a fully connected feed-forward artificial neural network completely in analog hardware. Part of Columbia ELENE3390, Electrical Engineering Senior Capstone Projects

Best pre-finetuning model -- Train Loss: 0.7688, Val Loss: 0.2113, Val Accuracy: 100.00%
Best finetuning model -- Train Loss: 0.4129, Val Loss: 0.1293, Val Accuracy: 100.00%

## Realized MLP PCB

![IMG_8199](https://github.com/user-attachments/assets/262832ba-7dc2-440f-a816-e2f34c7856bc)

![IMG_8071](https://github.com/user-attachments/assets/2d557b08-689b-48be-8ebb-a837f0a84ec8)


## Modeling the Network in the Digital Domain


A 10x10 matrix created by converting a bitmap of the letter `A` to a binary matrix:
```
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 1 1 0 0 0 0]
 [0 0 0 1 1 1 0 0 0 0]
 [0 0 0 1 0 1 0 0 0 0]
 [0 0 1 1 1 1 1 0 0 0]
 [0 0 1 1 0 1 1 0 0 0]
 [0 0 1 1 0 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
```

And scaled to a 20x20 matrix:
```
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 0]
 [0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```

10x10 Matrix, 100 Epochs, Batch Size 64
| Hidden Layers | Training Loss | Validation Loss | Validation Accuracy |
|---------------|---------------|-----------------|---------------------|
| [2]           | 3.1931        | 3.3016          | 0.00%               |
| [2,2]         | 3.2458        | 3.2458          | 0.00%               |
| [4]           | 3.0239        | 2.8984          | 50.00%              |
| [4,4]         | 3.1435        | 3.1451          | 0.00%               |
| [8]           | 2.9434        | 2.8809          | 33.33%              |
| [8,8]         | 3.1306        | 2.9969          | 33.33%              |
| [16]          | 2.4503        | 2.3143          | 66.67%              |
| [16,16]       | 2.5845        | 2.495           | 66.67%              |
| [32]          | 2.0512        | 1.8625          | 100.00%             |
| [32,32]       | 1.6095        | 1.5628          | 66.67%              |
| [128,64]      | 0.1866        | 0.1601          | 100.00%             |


![Loss for n=[32]](images/10_10_8_1_loss.png)
![Accuracy for n=[32]](images/10_10_8_1_accuracy.png)
