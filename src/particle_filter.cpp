/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <cmath>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
	num_particles = 100;  // TODO: Set the number of particles
  	default_random_engine gen;

  	// This line creates a normal (Gaussian) distribution for x,y,Theta
  	normal_distribution<double> dist_x(x, std[0]);
  	normal_distribution<double> dist_y(y, std[1]);
  	normal_distribution<double> dist_theta(theta, std[2]);

  	// Sample from these normal distributions like this:
  	// sample_x = dist_x(gen);
  	// where "gen" is the random engine initialized earlier.
  	for (int index = 0; index < num_particles; index++)
  	{
        Particle particle;
		particle.id = index;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

    	particles.push_back(particle);
  		weights.push_back(particle.weight);
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
    default_random_engine gen;

  	for (int index = 0; index < num_particles; index++)
    {
    	double new_x, new_y, new_theta;
        if(yaw_rate == 0)
        {
        	new_x = particles[index].x + velocity * cos(particles[index].theta) * delta_t;
          	new_y = particles[index].y + velocity * sin(particles[index].theta) * delta_t;
          	new_theta = particles[index].theta;
        }
      	else
        {
        	new_x = particles[index].x + (velocity/yaw_rate) * (sin(particles[index].theta + yaw_rate * delta_t) - sin(particles[index].theta));
          	new_y = particles[index].y + (velocity/yaw_rate) * (-cos(particles[index].theta + yaw_rate * delta_t) + cos(particles[index].theta));
          	new_theta = particles[index].theta + (yaw_rate*delta_t);
        }

        normal_distribution<double> dist_x(new_x, std_pos[0]);
        normal_distribution<double> dist_y(new_y, std_pos[1]);
        normal_distribution<double> dist_theta(new_theta, std_pos[2]);

      	particles[index].x = dist_x(gen);
      	particles[index].y = dist_y(gen);
      	particles[index].theta = dist_theta(gen);
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
  	for (unsigned int index_obs = 0; index_obs < observations.size(); index_obs++)
  	{
    	double min_dist = numeric_limits<double>::max(); // Maximum finite value representable with double type
    	int ID_value = -1; // An ID that possibly cannot belong to a map landmark
    	for (int index_pred = 0; index_pred < predicted.size(); index_pred++)
    	{
      		double diff = dist(observations[index_obs].x, observations[index_obs].y, predicted[index_pred].x, predicted[index_pred].y);
      		if (diff < min_dist)
      		{
        		min_dist = diff;
        		ID_value = predicted[index_pred].id;
      		}
      		observations[index_obs].id = ID_value;
    	}
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
  	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
	
	for(int i = 0; i<particles.size(); i++){
		vector<LandmarkObs> t_obs(observations.size());
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		//transform to global CS
		for(int j = 0; j<observations.size(); j++){
			double xo = observations[j].x;
			double yo = observations[j].y;
			t_obs[j].x = x + xo*cos(theta) - yo*sin(theta);
			t_obs[j].y = y + xo*sin(theta) + yo*cos(theta);
			t_obs[j].id = observations[j].id;
		}
		
		vector<LandmarkObs> n_lm; //nearest Landmark
		
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];
			double xm = lm.x_f;
			double ym = lm.y_f;
			int idm = lm.id_i;
			
			double c_dist;
			c_dist = dist(x, y, xm, ym) ;
			
			if(c_dist<sensor_range){
				LandmarkObs pred_lm;
				pred_lm.id = idm;
				pred_lm.x = xm;
				pred_lm.y = ym;
				n_lm.push_back(pred_lm);
			}
			
		}
		
		dataAssociation(n_lm, t_obs);
		double t_weight;
		
		for(int j=0; j<t_obs.size(); j++){
			double x_obs = t_obs[j].x;
			double y_obs = t_obs[j].y;
			double mu_x;
			double mu_y;
			
			for (int k=0; k<n_lm.size(); k++){
				if(n_lm[k].id == t_obs[j].id){
					mu_x = n_lm[k].x;
					mu_y = n_lm[k].y;
				}
			}
			
			double exponent= pow((x_obs - mu_x),2)/(2 * sig_x*sig_x) + (pow((y_obs - mu_y),2)/(2 * sig_y*sig_y));
			t_weight= gauss_norm * exp(-exponent);
			
		}
		particles[i].weight = t_weight;
		weights[i] = t_weight;
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   	default_random_engine gen;     // Random engine initialized
	discrete_distribution<int> d_dist(weights.begin(), weights.end());
	vector<Particle> t_particles;
	for(int loop = 0; loop <particles.size(); loop++)
    {
		int d_particle = d_dist(gen);
		t_particles.push_back(particles[d_particle]);
		weights[loop] = particles[d_particle].weight;		
	}
	particles = t_particles;
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