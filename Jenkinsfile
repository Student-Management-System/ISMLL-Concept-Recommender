pipeline {
    agent any
    
    environment {
        DEMO_SERVER = '147.172.178.30'
    }
    
    stages {

        stage('Git') {
            steps {
                cleanWs()
                git 'https://github.com/Student-Management-System/ISMLL-Concept-Recommender.git'
            }
        }

        // Based on: https://medium.com/@mosheezderman/c51581cc783c
        stage('Deploy') {
            steps {
                sshagent(credentials: ['Stu-Mgmt_Demo-System']) {
                    sh """
                        # [ -d ~/.ssh ] || mkdir ~/.ssh && chmod 0700 ~/.ssh
                        # ssh-keyscan -t rsa,dsa example.com >> ~/.ssh/known_hosts
                        ssh -i ~/.ssh/id_rsa_student_mgmt_backend elscha@${env.DEMO_SERVER} <<EOF
                            cd ~/ISMLL-Concept-Recommender
                            git reset --hard
                            git pull
                            chmod +x demo.sh
                            systemctl --user restart recommender_api.service
                            exit
                        EOF"""
                }
            }
        }
    }
}