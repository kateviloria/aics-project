# Notes

Here is your wiki where you can document all stages of the project and save intermediate results and analyses.

### Logbook  
**Week 51**  
- Look for datasets
- Connect to past readings and tasks shown during tutorials
- Consult Simon on idea and possible improvements in the task  

**Week 52**  
- Write and submit project proposal

**Week 6**
- Set up GitHub repos with code from tutorial + download data on server
- Create pandas dataframe with annotations in JSON file and export as .csv file
- Create a written paper outline

**Week 7**
- Add Nikolai's image captioning code
- Make changes to fit dataframe and files
- Train encoder/decoder
- update READMEs
- Finalise annotations JSON file and pandas dataframe
- Update written paper outline

**Week 8**
- Beam search not working properly, decided to change to greedy
- Made greedy captioning code but not working due to GPU memory  

**Week 11**
- Retrying greedy caption code, still having problems with GPU

**Week 12**
- Ran greedy (for few images), results not great, need to reevaluate dataset
    - Model might be overfitting "quality issues are too severe to recognize visual content"
    - Part of captions that some annotators included when pictures were indistinct
