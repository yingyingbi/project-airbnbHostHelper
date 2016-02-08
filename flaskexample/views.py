import pygal
from pygal.style import Style
from pygal import Config

from flask import Flask, Response
from flask import request 
from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

#from ldaReviewModel_Test_USER import ldaReviewModel_Test_USER
#from ldaReviewModel_Test_LDA20_3 import ldaReviewModel_Test
from ldaReviewModel_Test_USER2 import ldaReviewModel_Test_USER2

user = 'postgres' #add your username here (same as previous postgreSQL)

host = 'localhost'
dbname = 'airbnbReview_db'
db = create_engine('postgres://%s@%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/input')
def hostHelper_input():
    return render_template("input.html")

@app.route('/output')
def hostHelper_output():
  number_of_dimension = 8
  selected_topics = [2,9,11,12,14,15,17,18]
  topic_dimension_table = {2:'DIMENSION_1',9:'DIMENSION_2',11:'DIMENSION_3',12:'DIMENSION_4',14:'DIMENSION_5',15:'DIMENSION_6',17:'DIMENSION_7',18:'DIMENSION_8'}
  ## pull 'listing_id' from input field and store it
  listing = request.args.get('Listing_ID')
  if not listing:
        return render_template("input.html")
  print listing

  ## zipcode query:
  zip_query = "SELECT ZIPCODE FROM listing_data_table WHERE id ='%s'"% listing
  zip_result = pd.read_sql_query(zip_query,con)
  if zip_result.empty:
       return render_template("error.html")
  zipcode = zip_result['zipcode'].values[0]
  
  ## review query:
  review_query = "SELECT * FROM review_data_table WHERE listing_id ='%s'"% listing
  review_result = pd.read_sql_query(review_query,con)

  
  raw_review_texts = []
  for i in range(0,review_result.shape[0]):
      raw_review_texts.append(review_result.iloc[i]['comments'])
    
  ## apply LDA model
  topic_result, prob_result, avg_score = ldaReviewModel_Test_USER2(raw_review_texts)
  
  ## Reference query
  ref_query = "SELECT * FROM listing_trained_table2 where zipcode ='%s' ORDER BY avg_score DESC, num_of_reviews DESC, review_score DESC LIMIT 1" %zipcode
  ref_result = pd.read_sql_query(ref_query,con)
  ref_listing_id = ref_result['id'].values[0]
  ref_score =[ref_result['dimension_1'].values[0],ref_result['dimension_2'].values[0],ref_result['dimension_3'].values[0],ref_result['dimension_4'].values[0],ref_result['dimension_5'].values[0],ref_result['dimension_6'].values[0],ref_result['dimension_7'].values[0],ref_result['dimension_8'].values[0]]
  
  ## reference review query:
  ref_review_query = "SELECT COMMENTS FROM review_trained_table2 where listing_id ='%s' ORDER BY (DIMENSION_1 + DIMENSION_2 + DIMENSION_3 + DIMENSION_4 + DIMENSION_5 + DIMENSION_6 + DIMENSION_7 + DIMENSION_8) DESC LIMIT 1"%ref_listing_id
  output_review = pd.read_sql_query(ref_review_query,con)
  output_review_text = output_review.values[0]
  
  ref_link = "https://www.airbnb.com/rooms/%s" % ref_listing_id
        
  ## generate chart
  custom_style = Style(
      background='transparent',
      plot_background='transparent',
      foreground='#000000',
      foreground_strong='#FF9900',
      foreground_subtle='#630C0D',
      opacity='.6',
      opacity_hover='.9',
      transition='400ms ease-in',
      legend_font_size = 18,
      label_font_size = 18,
      
      stroke_width = 10,
      font_family = "monospace",
      guide_stroke_width = 6,
      
      )
  
  radar_chart = pygal.Radar(dynamic_print_values=True, style=custom_style, width = 600, height = 600, stroke_style={'width': 4},dots_size = 4, legend_at_bottom = True, legend_at_bottom_columns = 1)
  
  radar_chart.x_labels = ['Communication', 'Transportation', 'Area:Convenience', 'Experience', 'Arrival', 'Area:Fun', 'Food', 'Amenity']
  radar_chart.add('Your Listing', prob_result, fill = True)
  #radar_chart.add('Listing '+str(listing), prob_result)
  #radar_chart.add('Listing '+str(ref_listing_id), ref_score)
  radar_chart.add('Best listing in your area', ref_score)  
  chart = radar_chart.render(is_unicode=True)
  

  return render_template("output.html", results = topic_result, output_reviews = output_review_text, ref = ref_listing_id, ref_link = ref_link, chart=chart)
















