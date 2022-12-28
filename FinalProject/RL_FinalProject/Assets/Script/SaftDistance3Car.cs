using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class SaftDistance3Car : Agent
{
    [SerializeField] private float currentSpeed;

    [SerializeField] private List<Vector3> BusiniPosition = new List<Vector3>();
    [SerializeField] private List<Transform> BusTransList;
    
    [SerializeField] private GameObject BusPrefab;

    [SerializeField] private Transform Goal;
    [SerializeField] private float distance;

    [SerializeField] private bool isInGoal;
    [SerializeField] private int goalCount;

    [Header("Camera")]
    [SerializeField] private Camera vehicleCam;

    [SerializeField] private List<Transform> YOLOdetectList;

    [SerializeField] private float moveSpeed = 1f;
    [SerializeField] private Transform targetTransform;
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;

    [SerializeField] private List<GameObject> exsistCar = new List<GameObject>();


    private void Start()
    {
        for (int n = 0; n < BusTransList.Count; n++)
        {
            BusiniPosition.Add(BusTransList[n].localPosition);
        }
    }

    public override void OnEpisodeBegin()
    {
        currentSpeed = Random.Range(40, 130); // 40, 130
        float currentSpeedPerSecond = currentSpeed * 1000 / 3600;

        // Set transform
        transform.localPosition = new Vector3(Random.Range(-85, -10), 0, 0); // -85, -10

        for (int n = 0; n < exsistCar.Count; n++)
        {
            Destroy(exsistCar[n]);
        }

        YOLOdetectList.Clear();
        exsistCar.Clear();

        for (int n = 0; n < BusTransList.Count; n++)
        {
            BusTransList[n].localPosition = new Vector3(Random.Range(10, 80), BusTransList[n].localPosition.y, BusiniPosition[n].z + Random.Range(-1f, 1f));

            if (n >= 1)
            {
                if (Random.Range(0f, 1f) > 0.4) //有機率生成車子
                {
                    GameObject go = Instantiate(BusPrefab, BusTransList[n].position, Quaternion.identity, transform.parent);
                    go.transform.Rotate(0, 90, 0);
                    exsistCar.Add(go);

                    YOLOdetectList.Add(go.transform.GetChild(0));
                    YOLOdetectList.Add(go.transform.GetChild(1));
                }
            }
            else
            {
                YOLOdetectList.Add(BusTransList[n].GetChild(0));
                YOLOdetectList.Add(BusTransList[n].GetChild(1));
            }
        }

        //BusTrans.localPosition = new Vector3(Random.Range(10, 80), 0, 0);

        distance = Function(currentSpeedPerSecond);
        Goal.localPosition = new Vector3(0, 0, -1 * distance);

        isInGoal = false;
        goalCount = 0;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        //sensor.AddObservation(transform.position);
        //sensor.AddObservation(targetTransform.position);

        Vector2 positionLD = new Vector2(0, 0);
        Vector2 positionRU = new Vector2(0, 0);

        List<Vector2> positionList = new List<Vector2>();

        for (int n = 0; n < YOLOdetectList.Count; n += 2)
        {
            positionLD = vehicleCam.WorldToScreenPoint(YOLOdetectList[n].position);
            positionRU = vehicleCam.WorldToScreenPoint(YOLOdetectList[n + 1].position);

            positionList.Add(positionLD + (positionRU - positionLD) / 2);
        }

        while(positionList.Count < 3)
        {
            positionList.Add(Vector2.zero);
        }

        //List<int> ObservationList = new List<int>();

        //for (int n = 0; n< positionList.Count; n++)
        //{
        //    ObservationList.Add(n);
        //}

        //ObservationList = RandomChoiceList(ObservationList);

        for (int n = 0; n < positionList.Count; n++)
        {
            sensor.AddObservation(positionList[n]);
        }
        print("CurrentSpeed: " + currentSpeed + "CarPosition: " + positionList[0].ToString()+ positionList[1].ToString()+ positionList[2].ToString());
        sensor.AddObservation(currentSpeed);

    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Move The Car
        float moveX = actions.ContinuousActions[0];
        transform.position += new Vector3(moveX, 0, 0) * Time.deltaTime * moveSpeed;

        // Add Reward
        if (isInGoal)
        {
            goalCount++;
            AddReward(0.005f);
        }
        else
        {
            AddReward(-100 / MaxStep);
        }
        if (goalCount > 200)
        {
            SetReward(2f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> contionousActions = actionsOut.ContinuousActions;
        contionousActions[0] = Input.GetAxisRaw("Horizontal");
    }


    private float GetDistance()
    {
        return Vector3.Distance(BusTransList[0].position, vehicleCam.gameObject.transform.position);
    }

    private float Function(float currentSpeed)
    {
        return (float)(currentSpeed * currentSpeed / (2 * 9.8 * 0.8));
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-1f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
    }
    private void OnTriggerStay(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            isInGoal = true;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            isInGoal = false;
        }
    }

    List<int> RandomChoiceList(List<int> input)
    {
        List<int> outputList = new List<int>();

        for ( int n =0; n < input.Count; n++)
        {
            int choiceNumber = Random.Range(0, input.Count);

            while (outputList.Contains(choiceNumber))
            {
                choiceNumber = Random.Range(0, input.Count);
            }
            outputList.Add(choiceNumber);
        }

        return outputList;

    }
}
