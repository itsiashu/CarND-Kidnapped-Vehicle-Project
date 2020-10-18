/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

//using std::string;
//using std::vector;
//using std::normal_distribution
using namespace std;
// declare a random static engine to be used across project
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if(is_initialized) {
   return;
  }
  
  num_particles = 100;  // TODO: Set the number of particles
  // define normal distributions for sensor noise
  normal_distribution<double> N_x_init(x, std[0]);
  normal_distribution<double> N_y_init(y, std[1]);
  normal_distribution<double> N_theta_init(theta, std[2]);
  
  // initialize particles
  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = N_x_init(gen);
    p.y = N_y_init(gen);
    p.theta = N_theta_init(gen);
    p.weight = 1.0;
    // add noise
    //p.x += N_x_init(gen);
    //p.y += N_y_init(gen);
    //p.theta += N_theta_init(gen);
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // define normal distributions for sensor noise
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);
  
  for(int i = 0; i < num_particles; i++) {
   // calc new state
    if(fabs(yaw_rate) < 0.00001) { 
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // add noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(unsigned int i = 0; i < observations.size(); i++) {
    // grab current observation
    LandmarkObs obs = observations[i];
    
    // init minimum distance
    double min_dist = numeric_limits<double>::max();
    
    // initialize landmark id with map to be associated with the observation
    int map_id = -1;
    
    for(unsigned int j = 0; j < predicted.size(); j++) {
      // get current prediction
      LandmarkObs pred = predicted[j];
      
      // get distance between both
      double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
      
      // find predicted landmark closest to observed landmark
      if(cur_dist < min_dist) {
       min_dist = cur_dist;
       map_id = pred.id;
      }
    }
    // set observation id to closest predicted landmark's id
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(int i = 0; i < num_particles; i++) {
   double particle_x = particles[i].x;
   double particle_y = particles[i].y;
   double particle_theta = particles[i].theta;
   
   // setup a predictions landmark vector
   vector<LandmarkObs> predictions;
    
   for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
     // fetch id and x, y values
     float lm_x = map_landmarks.landmark_list[j].x_f;
     float lm_y = map_landmarks.landmark_list[j].y_f;
     int lm_id = map_landmarks.landmark_list[j].id_i;
     
     
     // consider landmarks only within particle's sensor range
     if (fabs(lm_x - particle_x) <= sensor_range && fabs(lm_y - particle_y) <= sensor_range) {
       // add predictions to vector
       predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
     }
   }
   // observations transformed from vehicle co-ordinates system to map co-ordinate system
   vector<LandmarkObs> transformed_obs;
   for (unsigned int j = 0; j < observations.size(); j++) {
     double t_x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
     double t_y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;
     transformed_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
   }
    
   // data association on current particle for the predictions and transformed observation
   dataAssociation(predictions, transformed_obs);
   
   // reinit weight
   particles[i].weight = 1.0;

   for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      // placeholders for observation and associated prediction coordinates
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_obs[j].x;
      obs_y = transformed_obs[j].y;

      int associated_prediction = transformed_obs[j].id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
        }
      }
      // calculate weight for this observation with multivariate Gaussian
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2))) ) );

      // multiply current obs weight with total observations weight
      particles[i].weight *= obs_w;
    } 
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  
  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;
  // resample
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}