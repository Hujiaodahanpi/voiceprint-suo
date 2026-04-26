from mvector.predict import MVectorPredictor

predictor = MVectorPredictor(configs='VoiceprintRecognition_Pytorch_develop/configs/cam++.yml',
                             model_path='VoiceprintRecognition_Pytorch_develop/models/训练其他超大数据集的模型/CAMPPlus训练其他超大数据集/models/CAMPPlus_Fbank/best_model',
                             audio_db_path='VoiceprintRecognition_Pytorch_develop/audio_db/',
                             threshold=0.6)
# # 获取音频特征
# embedding = predictor.predict(audio_data='dataset/a_1.wav')
# # 获取两个音频的相似度
# similarity = predictor.contrast(
#     audio_data1='VoiceprintRecognition_Pytorch_develop/dataset/0.wav', audio_data2='VoiceprintRecognition_Pytorch_develop/dataset/test1.wav')
# print(str(similarity))

# # 注册用户音频
# predictor.register(user_name='夜雨飘零', audio_data='dataset/test.wav')
# # 识别用户音频
name, score = predictor.recognition(audio_data='VoiceprintRecognition_Pytorch_develop/dataset/0.wav')
print("name: " + str(name))
print("score: " + str(score))
# # 获取所有用户
# users_name = predictor.get_users()
# # 删除用户音频
# predictor.remove_user(user_name='夜雨飘零')

# name, score = predictor.recognition(audio_data='VoiceprintRecognition_Pytorch_develop/dataset/test1.wav')
# print(name, score)
