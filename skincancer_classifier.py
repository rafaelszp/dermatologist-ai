from utils.improc import SkinCancerImageGenerator


prediction = {
    'train':{
        'dir': './data/train'
    }
}
train_gen = SkinCancerImageGenerator(prediction['train']['dir'])
teste = train_gen.__getitem__(30)
print(teste)

